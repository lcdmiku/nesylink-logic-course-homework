from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
import numpy as np

from vision_exact import Pos,pxPos,SymbolicObs,PixelPerception

# # 动作编号
from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_NOOP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN
)

# 符号 tile 编码，先和文档里的 grid code 保持一致
from vision_exact import (
    EMPTY,
    WALL,
    PLAYER,
    MONSTER,
    CHEST,
    EXIT,
    TRAP,
    BUTTON ,
    NPC,
    GAP,
    BRIDGE,
    SWITCH,
    #gird's metadata
    TILE_SIZE ,
    ROOM_W ,
    ROOM_H,
)

@dataclass
class BeliefState:
    task_id: Optional[str] = None
    step: int = 0

    # 当前房间记忆
    last_player: Optional[Pos] = None
    facing: str = "up"

    # 任务进度记忆
    has_key: bool = False
    has_sword: bool = False

    keys: int = 0
    gold: int = 0
    items: Set[str] = field(default_factory=set)
    tools: Set[str] = field(default_factory=set)
    
    opened_chests: Set[Pos] = field(default_factory=set)
    killed_monsters: Set[Pos] = field(default_factory=set)
    pressed_buttons: Set[Pos] = field(default_factory=set)

    # 失败检测
    last_action: int = ACTION_NOOP
    stuck_count: int = 0

    def reset(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.step = 0
        self.last_player = None
        self.facing = "up"
        self.has_key = False
        self.has_sword = False
        self.keys = 0
        self.gold = 0
        self.items.clear()
        self.tools.clear()
        self.opened_chests.clear()
        self.killed_monsters.clear()
        self.pressed_buttons.clear()
        self.last_action = ACTION_NOOP
        self.stuck_count = 0

    def update(self, sym: SymbolicObs, info=None):
        self.step += 1

        # 只把 info 当作兼容接口。最终不要读隐藏状态。
        # 目前允许谨慎读取 inventory，因为项目说明中物品栏可作为显式输入。
        inv = None
        if isinstance(info, dict):
            inv = info.get("inventory", None)

        if inv:
            old_keys = self.keys
            old_gold = self.gold
            old_items = set(self.items)
            old_tools = set(self.tools)

            self.keys = int(inv.get("keys", 0))
            self.gold = int(inv.get("gold", 0))
            self.items = set(inv.get("items", []))
            self.tools = set(inv.get("tools", []))

            self.has_key = self.keys > 0
            self.has_sword = ("sword" in self.tools) or ("sword" in self.items) or (inv.get("equipped", {}).get("A") == "sword")

            if self.keys > old_keys:
                print(f"[LOOT] step={self.step} got KEY: {old_keys} -> {self.keys}")

            if self.gold > old_gold:
                print(f"[LOOT] step={self.step} got GOLD: {old_gold} -> {self.gold}")

            new_items = self.items - old_items
            if new_items:
                print(f"[LOOT] step={self.step} got ITEM: {new_items}")

            new_tools = self.tools - old_tools
            if new_tools:
                print(f"[LOOT] step={self.step} got TOOL: {new_tools}")

        # 更新 facing：根据玩家 tile 变化推断
        if self.last_player is not None and sym.player is not None:
            lx, ly = self.last_player
            x, y = sym.player
            if x > lx:
                self.facing = "right"
            elif x < lx:
                self.facing = "left"
            elif y > ly:
                self.facing = "down"
            elif y < ly:
                self.facing = "up"

            # 卡住检测
            if sym.player == self.last_player and self.last_action in {
                ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
            }:
                self.stuck_count += 1
            else:
                self.stuck_count = 0

        self.last_player = sym.player
        sym.facing = self.facing

        if isinstance(info, dict):
            events = info.get("events", {})
            flags = events.get("flags", {}) if isinstance(events, dict) else {}
            details = events.get("details", []) if isinstance(events, dict) else []

            interesting = [
                "chest_opened",
                "key_collected",
                "gold_collected",
                "item_collected",
                "agent_healed",
                "door_opened",
                "room_changed",
                "world_completed",
            ]

            happened = [name for name in interesting if flags.get(name, False)]

            if happened:
                print(
                    f"[EVENT] step={self.step}",
                    "happened=", happened,
                    "details=", details,
                )


@dataclass
class Subgoal:
    kind: str
    target: Optional[Pos] = None
    facing : Optional[int] = None


TASK_MILESTONES: dict[str, tuple[str, ...]] = {
    "mathematical_logic/task_3": (
        "monster_killed",
        "key_collected",
    ),
    "mathematical_logic/task_4": (
        "switch_activated",
        "key_collected",
        "door_opened",
        "item_collected",
        "monster_killed",
    ),
}

TASK5_EVENTS = (
    "chest_opened",
    "key_collected",
    "gold_collected",
    "item_collected",
    "agent_healed",
    "button_pressed",
    "room_changed",
    "door_opened",
    "trap_triggered",
    "monster_killed",
    "exit_reached",
    "environment_completed",
    "world_completed",
)

def neighbors(p: Pos) -> List[Tuple[Pos, int]]:
    x, y = p
    return [
        ((x, y - 1), ACTION_UP),
        ((x, y + 1), ACTION_DOWN),
        ((x - 1, y), ACTION_LEFT),
        ((x + 1, y), ACTION_RIGHT),
    ]

def nearest(start: Pos, candidates: List[Pos]) -> Optional[Pos]:
    """从candidates中找到距离start最近的一个"""
    if not candidates:
        return None
    sx, sy = start
    return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

def nearest_px(start: pxPos , candidates: List[pxPos]) -> Optional[pxPos]:
    """nearest函数的像素级别版本"""
    if not candidates:
        return None
    sx, sy = start
    return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))


def in_bounds(p: Pos) -> bool:
    """判断是否在grid合法范围内"""
    x, y = p
    return 0 <= x < ROOM_W and 0 <= y < ROOM_H


def is_passable(tile: int) -> bool:
    """判断tile能否通过，宝箱、墙、怪物、陷阱、gap 暂时都不走"""
    return tile in {EMPTY, PLAYER, EXIT, BUTTON, BRIDGE, SWITCH}

def is_monster(tile : int) -> bool:
    """判断是否是tile is monster"""
    return tile == MONSTER

def bfs_path(grid: np.ndarray, start: Pos, goal: Pos) -> List[int]:
    """
    返回 tile 级动作序列，比如 [RIGHT, RIGHT, UP]
    """
    if start == goal:
        return []

    q = deque([start])
    parent: Dict[Pos, Tuple[Optional[Pos], Optional[int]]] = {
        start: (None, None)
    }

    while q:
        cur = q.popleft()

        for nxt, act in neighbors(cur):
            if not in_bounds(nxt):
                continue
            if nxt in parent:
                continue

            x, y = nxt

            #lcd : 无视monster
            if (not is_passable(int(grid[y, x]))) and (not is_monster(int(grid[y,x]))):
                continue

            parent[nxt] = (cur, act)

            if nxt == goal:
                # 回溯动作
                actions = []
                p = nxt
                while parent[p][0] is not None:
                    prev, a = parent[p]
                    actions.append(a)
                    p = prev
                actions.reverse()
                return actions

            q.append(nxt)

    return []


def repeat_action(action: int, n: int) -> List[int]:
    return [action] * n


def expand_tile_actions(tile_actions: List[int]) -> List[int]:
    """将tile级别移动转化为pixel也就是像素级别pixel_actions"""
    pixel_actions = []
    for a in tile_actions:
        pixel_actions.extend(repeat_action(a, TILE_SIZE))
    return pixel_actions


def adjacent_tiles(pos: Pos) -> List[Pos]:
    """return tiles : List[Pos] adjacent to pos : Pos"""
    x, y = pos
    return [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]


def action_to_face(src: Pos, dst: Pos) -> int:
    """根据目标位置返回行动方向"""
    sx, sy = src
    dx, dy = dst
    if dx > sx:
        return ACTION_RIGHT
    if dx < sx:
        return ACTION_LEFT
    if dy > sy:
        return ACTION_DOWN
    if dy < sy:
        return ACTION_UP
    return ACTION_NOOP

def Str2Enum_facing(facing : str) -> int:
    """建立方向facing(str)与action(int)之间的转换"""
    relation = {
        "up":ACTION_UP,
        "down":ACTION_DOWN,
        "left":ACTION_LEFT,
        "right":ACTION_RIGHT
    }
    return relation.get(facing, ACTION_NOOP)

def is_encounter_monster(sym : SymbolicObs,bound = 10) -> int | None :
    """
    判断是否遭遇monster，如果是，返回monster所在方向facing,如果否，返回ACTION_NOOP=0
    遭遇是指player与monster距离近(bound)，可以直接朝facing方向进行攻击
    """
    player_px = sym.player_px
    monster_px = nearest_px(player_px,sym.monsters_px)

    if monster_px is None:
        return ACTION_NOOP
    #monster位于right
    if (monster_px[0] < player_px[0] + TILE_SIZE + bound) and (monster_px[0] > player_px[0] + TILE_SIZE) and \
            (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
        print(monster_px[0] , player_px[0] + TILE_SIZE + bound)

        return ACTION_RIGHT

    #monster位于left
    if (monster_px[0] + TILE_SIZE > player_px[0] - bound) and (monster_px[0] + TILE_SIZE < player_px[0]) and \
            (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
        return ACTION_LEFT

    #monster位于up
    if (monster_px[1] + TILE_SIZE > player_px[1] - bound) and (monster_px[1] + TILE_SIZE < player_px[1]) and \
            (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
        return ACTION_UP

    #monster位于down
    if (monster_px[1] < player_px[1] + TILE_SIZE+ bound) and (monster_px[1] > player_px[1] + TILE_SIZE) and\
            (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
        return ACTION_DOWN

    return ACTION_NOOP


class SymbolicPlanner:
    def next_subgoal(self, sym: SymbolicObs, belief: BeliefState) -> Subgoal:
        """
        上层 planner：决定现在应该干什么。
        先实现 Task 1/2/3通用逻辑：
        1. detect_near_monster -> hit_monster
        1. detect_chest_unopened -> find_chest
        2. have_key_and_detect_closedExit -> openExit_leave
        3. detect_normal_opened_exit -> leave
        """


        # 玩家位置识别失败时，不要乱动
        if sym.player is None:
            return Subgoal("wait")

        # 附近有monster
        monster_facing = is_encounter_monster(sym)
        if monster_facing:
            return Subgoal("kill_monster",facing=monster_facing)

        # 没钥匙：优先去最近宝箱 detect_chest_unopened -> open_chest
        # if not belief.has_key:
        #     chest = self.nearest(sym.player, sym.chests)
        #     if chest is not None:
        #         return Subgoal("find_chest", chest)
        #     return Subgoal("explore")
        chest = self.nearest(sym.player,sym.chests)
        if chest is not None:
            return Subgoal("find_chest",chest)

        # 有钥匙：去出口
        if belief.has_key:
            exit_pos = self.nearest(sym.player, sym.exits)
            if exit_pos is not None:
                return Subgoal("go_exit", exit_pos)

        return Subgoal("explore")

    def nearest(self, start: Pos, candidates: List[Pos]) -> Optional[Pos]:
        """从candidates中找到距离start最近的一个"""
        if not candidates:
            return None
        sx, sy = start
        return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))


class OptionController:
    def build_actions(
        self,
        sym: SymbolicObs,
        belief: BeliefState,
        subgoal: Subgoal
    ) -> List[int]:
        """根据子目标sub_goal返回actions列表"""
        if sym.player is None:
            return [ACTION_NOOP]

        if subgoal.kind == "wait":
            return [ACTION_NOOP]

        if subgoal.kind == "find_chest" and subgoal.target is not None:
            return self.actions_to_interactable(sym, subgoal.target)

        if subgoal.kind == "kill_monster" and subgoal.facing is not None:
            return self.actions_to_kill_monster(sym, subgoal.facing)

        if subgoal.kind == "go_exit" and subgoal.target is not None:
            return self.actions_to_exit(sym, subgoal.target)

        if subgoal.kind == "explore":
            # 最简单探索：先等一下，后面再做 frontier exploration
            return [ACTION_NOOP]

        return [ACTION_NOOP]

    def actions_to_kill_monster(self,sym: SymbolicObs, facing: int) -> List[int]:
        """
        朝指定方向进行攻击
        面向monster并攻击
        """
        assert sym.player is not None
        actions = []
        if facing != ACTION_NOOP and facing != Str2Enum_facing(sym.facing):
            actions.append(facing)
        # 按 A
        actions.append(ACTION_A)
        return actions

    def actions_to_exit(self, sym: SymbolicObs, exit_pos: Pos) -> List[int]:
        """获取前往exit的actions"""
        assert sym.player is not None

        # 先走到出口 tile
        tile_actions = bfs_path(sym.grid, sym.player, exit_pos)
        actions = expand_tile_actions(tile_actions)

        # 再朝边界方向多走 32 步
        out_action = self.exit_direction_from_tile(exit_pos)
        if out_action != ACTION_NOOP:
            actions.extend([out_action] * 32)

        return actions

    def exit_direction_from_tile(self, exit_pos: Pos) -> int:
        """确定exit所在方向"""
        x, y = exit_pos
        if y == 0:
            return ACTION_UP
        if y == ROOM_H - 1:
            return ACTION_DOWN
        if x == 0:
            return ACTION_LEFT
        if x == ROOM_W - 1:
            return ACTION_RIGHT
        return ACTION_NOOP

    def actions_to_interactable(self, sym: SymbolicObs, obj_pos: Pos) -> List[int]:
        """
        去到物体相邻格，然后面向物体，按 A。
        适用于 chest / switch / NPC
        """
        assert sym.player is not None

        candidates = []
        for p in adjacent_tiles(obj_pos):
            if not in_bounds(p):
                continue
            x, y = p
            if is_passable(int(sym.grid[y, x])):
                candidates.append(p)

        if not candidates:
            return [ACTION_NOOP]

        # 选距离玩家最近的相邻格
        px, py = sym.player
        target_adj = min(
            candidates,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py)
        )

        tile_actions = bfs_path(sym.grid, sym.player, target_adj)
        actions = expand_tile_actions(tile_actions)

        # 到达相邻格后，如果角色朝向不对，移动一步方向键让角色朝向宝箱
        face_action = action_to_face(target_adj, obj_pos)
        if face_action != ACTION_NOOP and face_action != sym.facing:
            actions.append(face_action)

        # 按 A
        actions.append(ACTION_A)
        return actions


class SafetyShield:
    def filter(self, action: int, sym: SymbolicObs, belief: BeliefState) -> int:
        """判断action是否合法，合法返回原action，否则返回wait"""
        if sym.player is None:
            return ACTION_NOOP

        if action in {ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT}:
            if self.is_exit_leaving_action(sym.player, action, sym.exits):
                return action

            nxt = self.predict_next_tile(sym.player, action)

            if not in_bounds(nxt):
                return ACTION_NOOP

            x, y = nxt
            tile = int(sym.grid[y, x])

            # 不主动走进墙、陷阱、gap、怪物
            if tile in {WALL, TRAP, GAP, MONSTER}:
                return ACTION_NOOP

        return action

    def predict_next_tile(self, pos: Pos, action: int) -> Pos:
        """预测沿着当前action方向player的下一个tile位置"""
        x, y = pos
        if action == ACTION_UP:
            return (x, y - 1)
        if action == ACTION_DOWN:
            return (x, y + 1)
        if action == ACTION_LEFT:
            return (x - 1, y)
        if action == ACTION_RIGHT:
            return (x + 1, y)
        return pos
    
    def is_exit_leaving_action(self, pos: Pos, action: int, exits: List[Pos]) -> bool:
        """判断是否到达exit并有离开的action"""
        x, y = pos
        
        if pos not in exits:
            return False

        return (
            (y == 0 and action == ACTION_UP) or
            (y == ROOM_H - 1 and action == ACTION_DOWN) or
            (x == 0 and action == ACTION_LEFT) or
            (x == ROOM_W - 1 and action == ACTION_RIGHT)
        )
    
class Policy:
    def __init__(self) -> None:
        self.perception = PixelPerception()
        self.belief = BeliefState()
        self.planner = SymbolicPlanner()
        self.controller = OptionController()
        self.shield = SafetyShield()

        self.action_queue: Deque[int] = deque()
        self.current_subgoal: Optional[Subgoal] = None

        self.last_sym: Optional[SymbolicObs] = None
        self.perception_interval = 4   # 先用 4，稳定后可以改成 8

        self.force_exit_action: Optional[int] = None
        self.force_exit_steps: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.belief.reset(task_id=task_id)
        self.action_queue.clear()
        self.current_subgoal = None

        self.last_sym = None
        self.force_exit_action = None
        self.force_exit_steps = 0
    
    def act(self, obs, info=None) -> int:

        # 已经进入强制出门模式：不要识图，不要 shield，直接往外走
        if self.force_exit_steps > 0 and self.force_exit_action is not None:
            self.force_exit_steps -= 1

            if self.force_exit_steps % 10 == 0:
                print(
                    "[FORCE_EXIT]",
                    "action=", self.force_exit_action,
                    "steps_left=", self.force_exit_steps,
                )

            self.belief.last_action = self.force_exit_action

            return int(self.force_exit_action)
        
        need_vision = (
        self.last_sym is None
        or not self.action_queue
        or self.belief.step % self.perception_interval == 0
        or self.last_sym.monsters  #如果有monster要继续vision
        )

        if need_vision:
            sym = self.perception(obs)
            self.last_sym = sym
            self.belief.update(sym, info)
        else:
            sym = self.last_sym
            self.belief.step += 1

        replanned = False
        # 附近有monster
        encounter_monster = is_encounter_monster(sym,)
        if encounter_monster:
            replanned = True

        # 3. 判断是否需要重新规划
        if self.need_replan(sym, info,replanned):
            replanned = True
            self.current_subgoal = self.planner.next_subgoal(sym, self.belief)
            actions = self.controller.build_actions(
                sym,
                self.belief,
                self.current_subgoal
            )
            self.action_queue = deque(actions)

            # 调试输出 1：只在重新规划时打印
            print(
                "[REPLAN]",
                "step=", self.belief.step,
                "player=", sym.player,
                "chests=", sym.chests,
                "exits=", sym.exits,
                "monsters=", sym.monsters,
                "has_key=", self.belief.has_key,
                "subgoal=", self.current_subgoal,
                "new_queue=", len(self.action_queue),
            )

        # 4. 没动作就等待
        if not self.action_queue:
            raw_action = ACTION_NOOP
        else:
            raw_action = self.action_queue.popleft()


        if (
            self.current_subgoal is not None
            and self.current_subgoal.kind == "go_exit"
            and self.is_border_leaving_action(sym.player, raw_action)
        ):
            self.force_exit_action = raw_action
            self.force_exit_steps = 40

            print(
                "[START_FORCE_EXIT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "raw=", raw_action,
                "exits=", sym.exits,
            )

            self.belief.last_action = raw_action
            return int(raw_action)


        # 5. 安全过滤
        action = self.shield.filter(raw_action, sym, self.belief)

        if self.belief.step % 10 == 0 or replanned:
            #lcd : 增加调试信息
            print(f"<<<<<<<<<<<< step: {self.belief.step} >>>>>>>>>>>>")
            print(
                "[INFO]",
                "player=", info['agent'],
                "monster=", info["entities"]["monsters_remaining"],
                "event=",info["events"]["records"]
            )
            print(
                "[ACT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "monster=",sym.monsters,
                "monster_px=",sym.monsters_px,
                "raw=", raw_action,
                "safe=", action,
                "queue_left=", len(self.action_queue),
                "stuck=", self.belief.stuck_count,
                "subgoal=", self.current_subgoal,
            )

        self.belief.last_action = action
        return int(action)

    def need_replan(self, sym: SymbolicObs, info=None,force_replan=False) -> bool:
        """判断是否需要重新规划"""
        #强制replan
        if force_replan:
            return True

        # 没有动作了，必须重新规划
        if not self.action_queue:
            return True

        # 识别不到玩家，先不继续盲走
        if sym.player is None:
            self.action_queue.clear()
            return True

        #发生了一些需要重新规划的事件


        # # 卡住了，重新规划
        # if self.belief.stuck_count >= 4:
        #     self.action_queue.clear()
        #     return True

        # # reward / info 里如果出现关键事件，也重新规划
        # # 注意：这里只建议用于训练/调试；最终要保证不使用隐藏状态。
        # if isinstance(info, dict):
        #     events = info.get("events", {})
        #     flags = events.get("flags", {}) if isinstance(events, dict) else {}

        #     important = [
        #         "chest_opened",
        #         "key_collected",
        #         "item_collected",
        #         "monster_killed",
        #         "door_opened",
        #         "button_pressed",
        #         "switch_activated",
        #         "bridge_rotated",
        #         "room_changed",
        #         "world_completed",
        #         "action_blocked",
        #     ]

        #     for name in important:
        #         if flags.get(name, False):
        #             self.action_queue.clear()
        #             return True

        return False
    
    def exit_action_if_at_exit(self, sym: SymbolicObs) -> Optional[int]:
        """
        如果玩家已经在出口边缘，返回应该朝哪个方向出门。
        允许出口检测只返回双格门的其中一个 tile。
        """
        if sym.player is None:
            return None

        px, py = sym.player

        for ex, ey in sym.exits:
            # north exit
            if ey == 0 and py == 0 and abs(px - ex) <= 1:
                return ACTION_UP

            # south exit
            if ey == ROOM_H - 1 and py == ROOM_H - 1 and abs(px - ex) <= 1:
                return ACTION_DOWN

            # west exit
            if ex == 0 and px == 0 and abs(py - ey) <= 1:
                return ACTION_LEFT

            # east exit
            if ex == ROOM_W - 1 and px == ROOM_W - 1 and abs(py - ey) <= 1:
                return ACTION_RIGHT

        return None

    def is_border_leaving_action(self, player: Optional[Pos], action: int) -> bool:
        """判断player位于border边缘且在撞墙"""
        if player is None:
            return False

        x, y = player

        return (
            (y == 0 and action == ACTION_UP) or
            (y == ROOM_H - 1 and action == ACTION_DOWN) or
            (x == 0 and action == ACTION_LEFT) or
            (x == ROOM_W - 1 and action == ACTION_RIGHT)
        )
    
def make_policy() -> Policy:
    return Policy()