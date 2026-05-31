import shutil

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from wrappers.syncode_wrapper import SyncodeLogitsProcessorWrapper
from trl import (
    ModelConfig,
    ScriptArguments)
from peft import get_peft_model
from rl_datasets.dataset_helper import BaseDatasetHelper
from rewards.base_reward import Reward
from wrappers.wrapper_utils import *

class RLOOWrapper(RLOOTrainer):
    def __init__(self, script_args : ScriptArguments, model_args : ModelConfig, rloo_args : RLOOConfig, dataset_helper : BaseDatasetHelper, reward_functions, cache_dir : str):

        self.script_args = script_args
        self.model_args = model_args

        self.subset_feature = dataset_helper.subset_feature
        self.full_dataset = dataset_helper.tokenized_dataset
        self.target_feature_name = dataset_helper.target_feature_name

        processed_dataset_for_model_generation = dataset_helper.preprocess_dataset()
        self.reward_class = Reward(target_language="python", reward_functions=reward_functions,
                                   dataset_name=dataset_helper.name) #ToDo target_language

        ref_policy, model, _, peft_config = load_model(cache_dir, model_args)
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        # Remove Output_dir if it exists
        shutil.rmtree(rloo_args.output_dir, ignore_errors=True)

        super().__init__(
            config=rloo_args,
            processing_class=dataset_helper.tokenizer,
            policy=model,
            ref_policy=ref_policy,
            reward_model=self.reward_class.calculate_reward,
            train_dataset=processed_dataset_for_model_generation["train"],
            eval_dataset=processed_dataset_for_model_generation["test"],
        )
        self.syncode_logits_processor = SyncodeLogitsProcessorWrapper(dataset_helper.tokenizer)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        self.syncode_logits_processor.do_sample = generation_config.do_sample

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                labels = data["labels"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                labels = labels.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []

                # Generate responses and compute logprobs
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss, syncode_scores = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                        [self.syncode_logits_processor]
                    )
                # Process responses in batches
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    label = labels[i : i + args.local_rollout_forward_batch_size]

                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    syncode_score = syncode_scores[i : i + args.local_rollout_forward_batch_size]

                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    attention_mask = syncode_score == 1.0
                    selected_output = postprocessed_response[attention_mask].unsqueeze(0)

                    """print("-----------------------------------------------------------")
                    print("Query:")
                    print(self._decode_batch(query)[0])
                    print("-----------------------------------------------------------")
                    print("Response:")
                    print(self._decode_batch(postprocessed_response)[0])
                    print("-----------------------------------------------------------")
                    print("Selected Response:")
                    print(self._decode_batch(selected_output)[0])
                    print("-----------------------------------------------------------")
                    print("syncode_score:")
                    print(syncode_score)
                    print("-----------------------------------------------------------")"""


                    if isinstance(reward_model, nn.Module):
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                    else:
                        self.reward_class.set_actual_target(decode_batch(label, processing_class))
                        score = torch.tensor(
                            reward_model(
                                decode_batch(selected_output, processing_class)
                            ),
                            dtype=torch.float,
                        ).to(device)

                    # Store batch results
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)

                # Concatenate all batched results
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, score)
                empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                # Compute KL divergence
                kl = logprobs - ref_logprobs

                # Normalize rewards
                if args.normalize_reward:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                # Compute total reward with KL penalty
                if args.token_level_kl:
                    # Token-level KL penalty: apply KL penalty per token
                    kl_reward = -args.kl_coef * kl

                    # Get the index of the last non-padded token for ea per_token_logpsch sequence
                    eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    last_reward = torch.zeros_like(kl)
                    # Ensure scores has correct shape and type
                    scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                    last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)

                    # Combine KL reward and last reward
                    non_score_reward = kl_reward.sum(1)  # Keep this for logging
                    reward = last_reward + kl_reward + syncode_scores.to(device)
                    rlhf_reward = reward.sum(1)  # Sum across sequence length
                else:
                    # Sequence-level KL penalty: sum KL across tokens first
                    sequence_kl = kl.sum(1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    rlhf_reward = non_score_reward + scores + syncode_scores.sum(1)

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()

                # Normalize advantages
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):

                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                        # Get batch data
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        # Forward pass
                        output = forward(model, mb_query_responses, processing_class.pad_token_id)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7

                        # Compute new logprobs
                        new_logprobs = selective_log_softmax(logits, mb_responses)
                        new_logprobs = torch.masked_fill(
                            new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                        )

                        # Compute probability ratios
                        new_ratio = (new_logprobs - mb_logprobs).exp()
                        new_logprobs = new_logprobs.sum(1)
                        mb_logprobs = mb_logprobs.sum(1)
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)

                        # PPO clipped loss
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = pg_loss_max.mean()

                        # Final loss
                        loss = pg_loss

                        # Optimization step
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        with torch.no_grad():
                            pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                pg_clipfrac
                            )
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    empty_cache()

            # Compute metrics
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_syncode_scores = syncode_scores.sum(1).mean().to(device)
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["objective/syncode_scores"] = self.accelerator.gather_for_metrics(mean_syncode_scores).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                labels = batch["labels"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _, syncode_scores = batch_generation(
                        unwrapped_model,
                        query,
                        1,
                        processing_class.pad_token_id,
                        generation_config,
                        [self.syncode_logits_processor]
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    attention_mask = syncode_scores == 1.0
                    selected_outputs = []
                    for mask, resp in zip(attention_mask, postprocessed_response):
                        selected_outputs.append(resp[mask])
                    selected_outputs = pad(selected_outputs, processing_class.pad_token_id, padding_side="right")

                    table["selected response"].extend(
                        gather_object(processing_class.batch_decode(selected_outputs))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                    if isinstance(self.reward_model, nn.Module):
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                    else:
                        self.reward_class.set_actual_target(decode_batch(labels, processing_class))
                        score = torch.tensor(
                            self.reward_model(
                                decode_batch(postprocessed_response, processing_class)
                            ),
                            dtype=torch.float,
                        ).to(postprocessed_response.device)
                    score += syncode_scores.sum(1).to(postprocessed_response.device)
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0: 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def test_generation(self):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        self.syncode_logits_processor.do_sample = generation_config.do_sample

        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response_mit_logit, _, syncode_scores = batch_generation(
                        unwrapped_model,
                        query,
                        1,
                        processing_class.pad_token_id,
                        generation_config,
                        [self.syncode_logits_processor],
                    )
                    response = query_response_mit_logit[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    query = decode_batch(query, processing_class)

                    attention_mask = syncode_scores == 1.0
                    selected_output = postprocessed_response[attention_mask].unsqueeze(0)
                    selected_output = decode_batch(selected_output, processing_class)

                    postprocessed_response = decode_batch(postprocessed_response, processing_class)
                    labels = decode_batch(batch["labels"], processing_class)
                    for idx in range(len(query)):
                        print("Query: ")
                        print(query[idx])
                        print("--------------------------------------------------------------------------------------------------------------------------")
                        print("mit logit")
                        print(postprocessed_response[idx])
                        print("--------------------------------------------------------------------------------------------------------------------------")
                        print("scores")
                        print(syncode_scores)
                        print(syncode_scores.shape[-1])
                        print("--------------------------------------------------------------------------------------------------------------------------")
                        print("selected output")
                        print(selected_output[idx])
                        print("--------------------------------------------------------------------------------------------------------------------------")
                        print("labels")
                        print(labels[idx])
                        print("--------------------------------------------------------------------------------------------------------------------------")
