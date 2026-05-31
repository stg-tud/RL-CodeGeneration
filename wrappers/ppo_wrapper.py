import shutil

from datasets import DatasetDict
from typing import Optional, List, Dict, Any, Union
from wrappers.wrapper_utils import *
from trl.trainer.ppo_trainer import *
from wrappers.syncode_wrapper import SyncodeLogitsListWrapper
from trl import (
    ModelConfig,
    PPOConfig,
    ScriptArguments)
from rl_datasets.dataset_helper import BaseDatasetHelper
from rewards.ruff_linter_reward import ruff_linter_reward

class PPOWrapper(PPOTrainer):
    def __init__(self, script_args : ScriptArguments, model_args : ModelConfig, ppo_args : PPOConfig, dataset_helper : BaseDatasetHelper,
                 framework_params, reward_functions, cache_dir : str, adapter_path : str = None):

        self.script_args = script_args
        self.model_args = model_args
        self.best_rlhf_reward = 0.0
        self.old_best_checkpoint = -1

        self.full_dataset = dataset_helper.tokenized_dataset
        self.target_feature_name = dataset_helper.target_feature_name
        self.model_input_size = dataset_helper.model_input_size

        self.reward_functions = reward_functions
        self.use_extra_rewards = framework_params['use_extra_rewards']
        self.syncode_coef = framework_params['syncode_coef']
        self.linter_coef = framework_params['linter_coef']
        self.rewards_coefs = framework_params['rewards_coef']

        if isinstance(dataset_helper.tokenized_dataset , (Dataset, DatasetDict)):
            processed_dataset_for_model_generation = dataset_helper.preprocess_dataset()
        else:
            processed_dataset_for_model_generation = dataset_helper.tokenized_dataset

        ref_policy, model, critic, peft_config = load_model(cache_dir, model_args, adapter_path)

        if adapter_path is None:
            # Remove Output_dir if it exists
            shutil.rmtree(ppo_args.output_dir, ignore_errors=True)

        collator = StringAdaptedDataCollator(dataset_helper.tokenizer, padding=True)

        super().__init__(
            args=ppo_args,
            processing_class=dataset_helper.tokenizer,
            model=model,
            ref_model=ref_policy,
            reward_model=critic,
            value_model=critic,
            peft_config=peft_config,
            train_dataset=processed_dataset_for_model_generation["train"],
            eval_dataset=processed_dataset_for_model_generation["test"],
            data_collator=collator
        )
        self.syncode_logits_processor = SyncodeLogitsListWrapper(dataset_helper.tokenizer, self.args.local_rollout_forward_batch_size)
        self.linter_function = ruff_linter_reward

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
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
            top_p=0.95,
            do_sample=True,
        )

        self.syncode_logits_processor.do_sample = generation_config.do_sample

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
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

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                labels = data["labels"].to(device)
                unit_tests = None
                if "unit_tests" in data:
                    unit_tests = data["unit_tests"]

                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []

                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss, syncode_scores = adjusted_batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                        [self.syncode_logits_processor],
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    syncode_score = syncode_scores[i: i + args.local_rollout_forward_batch_size]
                    label = labels[i: i + args.local_rollout_forward_batch_size]
                    if unit_tests is not None:
                        unit_test = unit_tests[i: i + args.local_rollout_forward_batch_size]

                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model

                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    score, _ = self.linter_function(
                            postprocessed_response,
                            processing_class
                        )
                    score = self.linter_coef * score.to(syncode_score.device) + self.syncode_coef * syncode_score

                    if unit_test is not None:
                        kwargs = {
                            "unit_tests": unit_test,
                        }
                    else:
                        kwargs = {}

                    if self.use_extra_rewards:
                        rewards = torch.zeros_like(score, dtype=torch.bfloat16).to(postprocessed_response.device)
                        for reward_function, reward_coef in zip(self.reward_functions, self.rewards_coefs):
                            rewards += reward_coef * reward_function(postprocessed_response, processing_class, score, label, **kwargs)

                        score = rewards.to(score.device) + score

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)


                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
                kl_reward = -args.kl_coef * kl
                rewards = kl_reward.clone()

                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                scores = scores.to(rewards.device)
                scores = torch.masked_fill(scores, padding_mask, 0.0)
                rewards = scores + rewards

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
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
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean() # Sum across sequence length
                mean_scores = scores.sum(1).mean().to(device) # Sum across sequence length
                advantages_mean = advantages.sum(1).mean().to(device)
                mean_entropy = (-logprobs).sum(1).mean() # Sum across sequence length
                mean_kl_reward = kl_reward.sum(1).mean() # Sum across sequence length
                rlhf_reward = rewards.sum(1).mean().to(device)
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/kl_reward"] = (
                    self.accelerator.gather_for_metrics(mean_kl_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/advantages"] = self.accelerator.gather_for_metrics(advantages_mean).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(mean_scores).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                #model.save_pretrained()
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_kl_reward, scores, metrics, kl_reward
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                rlhf_reward = self.generate_completions(sampling=True)
                self.accelerator.wait_for_everyone()
                # Create a tensor for the current reward on each rank
                current_reward_tensor = torch.tensor(rlhf_reward, device=self.accelerator.device)

                all_best_rewards = self.accelerator.gather_for_metrics(
                    current_reward_tensor.clone().detach()
                )
                global_synchronized_best_reward = all_best_rewards.max().item()
                is_new_best = global_synchronized_best_reward >= self.best_rlhf_reward
                print(f"Global Best Reward this step: {global_synchronized_best_reward}")
                if is_new_best:
                    if self.old_best_checkpoint > 0:
                        shutil.rmtree(args.output_dir + f"/checkpoint-{self.old_best_checkpoint}", ignore_errors=True)
                    self.old_best_checkpoint = self.state.global_step
                    self.best_rlhf_reward = global_synchronized_best_reward
                    print(f"New global best reward found! Saving checkpoint.")
                    self._save_checkpoint(model, trial=None)  # Rank 0 saves the model
                    self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            #model.save_pretrained()
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        self.generate_completions(sampling=True)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=0.95,
            do_sample=True
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                labels = batch["labels"]
                unit_tests = None
                if "unit_tests" in batch:
                    unit_tests = batch["unit_tests"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _, syncode_scores = adjusted_batch_generation(
                        unwrapped_model.policy,
                        query,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                        [self.syncode_logits_processor]
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["labels"].extend(
                        gather_object(processing_class.batch_decode(labels, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response, skip_special_tokens=True))
                    )

                    scores, _ = self.linter_function(
                            postprocessed_response,
                            processing_class
                        )

                    scores = self.linter_coef * scores.to(postprocessed_response.device) + self.syncode_coef * syncode_scores.to(postprocessed_response.device)

                    table["scores"].extend(self.accelerator.gather_for_metrics(scores).float().cpu().numpy())

                    end_score = scores
                    seprate_rewards = []
                    if unit_tests is not None:
                        kwargs = {
                            "unit_tests": unit_tests,
                            "is_eval": False,
                        }
                    else:
                        kwargs = {"is_eval": False}
                    if self.use_extra_rewards:
                        rewards = torch.zeros_like(scores, dtype=torch.bfloat16).to(postprocessed_response.device)
                        for reward_function, reward_coef in zip(self.reward_functions, self.rewards_coefs):
                            r = reward_coef * reward_function(postprocessed_response, processing_class, scores, labels, **kwargs)
                            seprate_rewards.append(r)
                            rewards += r

                        rlhf_reward = rewards.sum(1).mean().to(postprocessed_response.device)

                        for idx, r in enumerate(seprate_rewards):
                            table[f"reward_{idx}"].extend(
                                self.accelerator.gather_for_metrics(r).float().cpu().numpy()
                            )

                        end_score = rewards.to(postprocessed_response.device) + scores
                        table["end_score"].extend(self.accelerator.gather_for_metrics(end_score).float().cpu().numpy())

                    attention_mask = end_score > 0.0
                    attention_mask = attention_mask.to(postprocessed_response.device)
                    selected_output = torch.masked_fill(postprocessed_response, ~attention_mask,
                                                      processing_class.pad_token_id)

                    table["selected response"].extend(
                        gather_object(processing_class.batch_decode(selected_output, skip_special_tokens=True))
                    )
                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

        return rlhf_reward

    def generate_eval(self, prompts, batch_size, max_length):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=0.95,
            do_sample=True
        )
        queries = processing_class(prompts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)["input_ids"].to(self.accelerator.device)
        decoded_responses = []
        table = defaultdict(list)
        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    query = queries[i : i + batch_size]
                    context_length = query.shape[1]
                    query_response, _, syncode_scores = adjusted_batch_generation(
                        unwrapped_model.policy,
                        query,
                        batch_size,
                        processing_class.pad_token_id,
                        generation_config
                    )
                    responses = query_response[:, context_length:]
                    decoded_response = processing_class.batch_decode(responses, skip_special_tokens=True)

                    if self.accelerator.is_main_process:
                        df = pd.DataFrame(table)
                        table["queries"] = processing_class.batch_decode(query, skip_special_tokens=True)
                        table["model_response"] = decoded_response
                        decoded_responses.extend(decoded_response)
                        if is_rich_available():
                            print_rich_table(df.iloc[0: 0 + 10])
                        if "wandb" in args.report_to:
                            import wandb

                            if wandb.run is not None:
                                wandb.log({"completions": wandb.Table(dataframe=df)})

                        if "comet_ml" in args.report_to:
                            log_table_to_comet_experiment(
                                name="completions.csv",
                                table=df,
                            )
                        print(f"generation count {int(i / batch_size) + 1} / {int(len(prompts) / batch_size)} ")

        if self.accelerator.is_main_process:
            return decoded_responses, self.accelerator.is_main_process
        return None, False


    def save(self):
        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model
        self._save_checkpoint(self.model, trial=None)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        value_model = self.model.value_model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        output_dir_policy = os.path.join(output_dir, "policy")
        output_dir_value = os.path.join(output_dir, "value")

        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir_policy)
        value_model.save_pretrained(output_dir_value)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

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
                    query_response_mit_logit, _, syncode_scores = adjusted_batch_generation(
                        unwrapped_model.policy,
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
                    attention_mask = attention_mask.to(postprocessed_response.device)
                    selected_output = torch.masked_fill(postprocessed_response, ~attention_mask, processing_class.pad_token_id)
                    selected_output_decoded = decode_batch(selected_output, processing_class)

                    postprocessed_response_decoded = decode_batch(postprocessed_response, processing_class)
                    labels = batch["labels"]
                    labels_decoded = decode_batch(labels, processing_class)
                    for idx in range(len(query)):
                        print("Query: ")
                        print(query[idx])
                        print(
                            "--------------------------------------------------------------------------------------------------------------------------")
                        print("Response:")
                        print(postprocessed_response_decoded[idx])
                        print(
                            "--------------------------------------------------------------------------------------------------------------------------")
                        rewards, violations = self.linter_function([postprocessed_response[idx]], processing_class)
                        rewards = rewards.to(syncode_scores.device) + syncode_scores[idx]

                        attention_mask = rewards > 0.0
                        attention_mask = attention_mask.to(postprocessed_response.device)
                        select_output = torch.masked_fill(postprocessed_response, ~attention_mask,
                                                          processing_class.pad_token_id)
                        code_string = processing_class.decode(select_output[idx], skip_special_tokens=True)
                        print("code_string:")
                        print(code_string)
                        print(
                            "--------------------------------------------------------------------------------------------------------------------------")
                        print("labels: ")
                        print(labels_decoded[idx])
                        att = labels[idx] != processing_class.eos_token_id
                        test_reward = torch.where(att, 1.0, 0.0)
                        test = self.reward_functions[0]([postprocessed_response[idx]],rewards, processing_class)
                        print(test)
                        print(
                            "--------------------------------------------------------------------------------------------------------------------------")
