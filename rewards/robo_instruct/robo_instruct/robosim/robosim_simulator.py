from rewards.robo_instruct.roboeval.benchmark.rtl import SPECIAL_DELIM
from rewards.robo_instruct.roboeval.benchmark.simulator import State

from typing import List
import numpy as np
from enum import Enum

class Types(Enum):
  OBJECT= "object"
  PERSON= "person"
  LOCATION= "location"
  OBJECT_OR_PERSON= "object_or_person"

class RobotExecution:
  def __init__(self, state : State):
    self.state = state
    self.is_holding = None
    # constraint is populated in runtime to ensure consistency
    self.existence_constraints = {
      "exist": {},
      "non-exist": {}
    }
    self.type_constraints = {}
    self.is_in_room_last_check = {
      "location" : None,
      "count" : 0,
    }
  
  ##### Verify Constraints #####
  # ensure consistency: return 0 => non-exist, 1 => exist, 2 => unknown
  def check_existence_constraints(self, entity : str) -> int:
    current_loc = self.state.get_robot_location()
    if self.existence_constraints["exist"].get(current_loc):
      if entity in self.existence_constraints["exist"][current_loc]:
        return 1
      
    if self.existence_constraints["non-exist"].get(current_loc):
      if entity in self.existence_constraints["non-exist"][current_loc]:
        return 0

    return 2
  
  def add_existence_constraint(self, entity : str, should_exist : bool) -> None:
    current_loc = self.state.get_robot_location()
    if should_exist:
      if self.existence_constraints["exist"].get(current_loc):
          self.existence_constraints["exist"][current_loc].add(entity)
      else:
        self.existence_constraints["exist"][current_loc] = set()
        self.existence_constraints["exist"][current_loc].add(entity)
    else:
      if self.existence_constraints["non-exist"].get(current_loc):
          self.existence_constraints["non-exist"][current_loc].add(entity)
      else:
        self.existence_constraints["non-exist"][current_loc] = set()
        self.existence_constraints["non-exist"][current_loc].add(entity)
  
  def remove_existence_constraint(self, entity : str, exist_remove : bool) -> None:
    current_loc = self.state.get_robot_location()
    if exist_remove:
        self.existence_constraints["exist"][current_loc].remove(entity)
    else:
        self.existence_constraints["non-exist"][current_loc].remove(entity)

  def check_type_constraint(self, entity : str, type_name : str) -> bool:
    if not self.type_constraints.get(entity):
      return True
    elif self.type_constraints[entity] == type_name:
      return True
    elif type_name == Types.OBJECT_OR_PERSON:
      return self.type_constraints[entity] in [Types.OBJECT, Types.PERSON]
    elif type_name == Types.LOCATION or type_name == Types.PERSON: # Person referred to by location (special implicature case)
      return self.type_constraints[entity] in [Types.LOCATION, Types.PERSON]
    return False
    
  def add_type_constraint(self, entity : str, type_name : str) -> None:
    if not self.type_constraints.get(entity):
      self.type_constraints[entity] = type_name
    elif self.type_constraints[entity] == Types.OBJECT_OR_PERSON:
      self.type_constraints[entity] = type_name
  ##### Verify Constraints #####
  
  # Get the current location of the robot.
  def get_current_location(self) -> str :
    loc = self.state.get_robot_location()
    return loc

  # Get a list of all rooms in the house.
  def get_all_rooms(self) -> List[str] :
    rooms = self.state.get_all_locations()    
    return rooms
  
  def is_in_room(self, entity : str) -> bool:
    if type(entity) is not str:
      raise Exception("RobotIsInRoomError: entity must be a string")
    entity = entity.strip()
    if len(entity) == 0:
      raise Exception(f"RobotIsInRoomError: entity is empty")
    print(f"CheckEntity {SPECIAL_DELIM} {entity} {SPECIAL_DELIM} {None}", flush=True)
    # check constraints
    self.add_type_constraint(entity, Types.OBJECT_OR_PERSON) # special case: object or person
    if not self.check_type_constraint(entity, Types.OBJECT_OR_PERSON):
      raise Exception(f"RobotTypeError: entity expected to be an object or a person, but got {self.type_constraints[entity]}")
    res = self.check_existence_constraints(entity)
    # heuristic: if the same thing is called 15 times, we randomly sample a new value
    current_loc = self.state.get_robot_location()
    if current_loc != self.is_in_room_last_check["location"]:
      self.is_in_room_last_check["location"] = current_loc
      self.is_in_room_last_check["count"] = 1
    else:
      self.is_in_room_last_check["count"] += 1
    if res == 1:
      if self.is_in_room_last_check["count"] >= 15: # avoid edge case: infinite loop
        self.remove_existence_constraint(entity, exist_remove=True)
      return True 
    elif res == 0:
      if self.is_in_room_last_check["count"] >= 15: # avoid edge case: infinite loop
        self.remove_existence_constraint(entity, exist_remove=False)
      return False

    # randomize check
    choices = [True, False]
    random_choice = bool(np.random.choice(choices))
    if random_choice:
      self.add_existence_constraint(entity, should_exist=True)
    else:
      self.add_existence_constraint(entity, should_exist=False)
    return random_choice

  def go_to(self, location : str) -> None :
    if type(location) is not str:
      raise Exception("RobotGoToError: location must be a string")
    location = location.strip()
    if len(location) == 0:
      raise Exception(f"RobotGoToError: location is empty")
    try:
      self.state.robot_location = location 
    except:
      raise Exception(f"RobotGoToError: update_robot_location failed")
    # check constraints
    self.add_type_constraint(location, Types.LOCATION)
    if not self.check_type_constraint(location, Types.LOCATION):
      raise Exception(f"RobotTypeError: expects a location, but got {self.type_constraints[location]}")
    print(f"GoTo {SPECIAL_DELIM} {location} {SPECIAL_DELIM} {None}", flush=True)

  def ask(self, person : str, question : str, options: List[str]) -> str :
    if type(person) is not str:
      raise Exception("RobotAskError: person must be a string")
    person = person.strip()
    if type(question) is not str:
      raise Exception("RobotAskError: question must be a string")
    question = question.strip()
    if type(options) is not list:
      raise Exception("RobotAskError: options must be a list")
    if len(options) < 1:
      raise Exception(f"RobotAskError: no option provided")
    for option in options:
      if type(option) is not str:
        raise Exception(f"RobotAskError: option '{option}' is not a string")
      if len(option) == 0:
        raise Exception(f"RobotAskError: option is an empty string")
    if len(question) == 0:
      raise Exception(f"RobotAskError: question is an empty string")
    # check constraints
    self.add_type_constraint(person, Types.PERSON)
    if not self.check_type_constraint(person, Types.PERSON):
      raise Exception(f"RobotTypeError: expects a person, but got {self.type_constraints[person]}")
    current_loc = self.state.get_robot_location()
    res = self.check_existence_constraints(person)
    if res == 0:
      raise Exception(f"RobotAskError: person does not exist in room")
    elif res == 2:
      self.add_existence_constraint(person, should_exist=True)
    
    # random answering
    random_answer = np.random.choice(options)
    print(f"Ask {SPECIAL_DELIM} {question} {SPECIAL_DELIM} {options}", flush=True)
    return str(random_answer)
    
  def say(self, message : str) -> None :
    if type(message) is not str:
      raise Exception("RobotSayError: message must be a string")
    message = message.strip()
    print(f"Say {SPECIAL_DELIM} {message} {SPECIAL_DELIM} {None}", flush=True)

  def pick(self, obj: str) -> None:
    if type(obj) is not str:
      raise Exception("RobotPickError: obj must be a string")
    obj = obj.strip()
    if len(obj) == 0:
      raise Exception(f"RobotPickError: obj is empty")
    if self.is_holding is not None:
      raise Exception(f"RobotPickError: Robot is already holding '{obj}'")
    self.is_holding = obj

    # check constraints
    self.add_type_constraint(obj, Types.OBJECT)
    if not self.check_type_constraint(obj, Types.OBJECT):
      raise Exception(f"RobotTypeError: expects an object, but got {self.type_constraints[obj]}")
    current_loc = self.state.get_robot_location()
    res = self.check_existence_constraints(obj)
    if res == 0:
      raise Exception(f"RobotPickError: object does not exist in room")
    elif res == 1:
      # if already exist, make it unknown
      self.remove_existence_constraint(obj, exist_remove=True)

    print(f"Pick {SPECIAL_DELIM} {obj} {SPECIAL_DELIM} {None}", flush=True)

  def place(self, obj: str) -> None:
    if type(obj) is not str:
      raise Exception("RobotPlaceError: obj must be a string")
    obj = obj.strip()
    if len(obj) == 0:
      raise Exception(f"RobotPlaceError: obj is empty")
    if self.is_holding == None:
      raise Exception(f"RobotPlaceError: Robot is not currently holding '{obj}'")
    if self.is_holding != obj:
      raise Exception(f"RobotPlaceError: Robot is currently holding '{self.is_holding}' but it is not '{obj}'")
    self.is_holding = None

    # check constraints.
    self.add_type_constraint(obj, Types.OBJECT)
    if not self.check_type_constraint(obj, Types.OBJECT):
      raise Exception(f"RobotTypeError: expects an object, but got {self.type_constraints[obj]}")
    res = self.check_existence_constraints(obj)
    if res == 0:
      self.remove_existence_constraint(obj, exist_remove=False)
      self.add_existence_constraint(obj, should_exist=True)
    elif res == 2:
      self.add_existence_constraint(obj, should_exist=True)
    print(f"Place {SPECIAL_DELIM} {obj} {SPECIAL_DELIM} {None}", flush=True)
