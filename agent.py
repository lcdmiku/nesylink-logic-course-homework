from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
import numpy as np
from pathlib import Path
from PIL import Image
from vision_exact import StaticTileClassifier, PlayerDetector, MonsterDetector, ExitDetector, ExitInfo, Pos, pxPos
import heapq


#lcd : 将能够放到vision_exact.py的代码尽量放到通过导入来使用，不再重复定义
# from vision_exact import Pos,pxPos,SymbolicObs,PixelPerception

#lcd : 将这些枚举从其他文件中导入，不重复定义
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

MOVE_REPEAT = 20

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
    blocked_exits: Set[Pos] = field(default_factory=set)

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
        self.blocked_exits.clear()
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
    
    def update_facing_from_action(self, action: int):
        if action == ACTION_UP:
            self.facing = "up"
        elif action == ACTION_DOWN:
            self.facing = "down"
        elif action == ACTION_LEFT:
            self.facing = "left"
        elif action == ACTION_RIGHT:
            self.facing = "right"


@dataclass
class SymbolicObs:
    grid: np.ndarray                    # shape: (8, 10)
    player: Optional[Pos] = None
    player_px: Optional[pxPos] = None
    facing: str = "up"
    monsters: List[Pos] = field(default_factory=list)
    chests: List[Pos] = field(default_factory=list)
    exits: List[Pos] = field(default_factory=list)

    exit_infos: List['ExitInfo'] = field(default_factory=list)
    exit_types: Dict[Pos, str] = field(default_factory=dict)
    exit_opened: Dict[Pos, bool] = field(default_factory=dict)
    traps: List[Pos] = field(default_factory=list)
    buttons: List[Pos] = field(default_factory=list)
    switches: List[Pos] = field(default_factory=list)


class PixelPerception:
    def __init__(self):
        self.static_clf = StaticTileClassifier()
        self.player_detector = PlayerDetector()
        self.monster_detector = MonsterDetector()
        self.exit_detector = ExitDetector()

    def __call__(self, obs):
        frame = np.asarray(obs)

        # 防止误传 render()，只取地图区域
        frame = frame[:128, :160, :3]

        grid = np.zeros((8, 10), dtype=np.int64)

        # 1. 静态 tile 分类
        for y in range(8):
            for x in range(10):
                patch = frame[y*16:(y+1)*16, x*16:(x+1)*16, :]
                label, name, score = self.static_clf.classify_tile(patch)

                # 这里阈值可以设严格一点。
                # 如果 score 太大，默认 floor，避免误判。
                if score < 500:
                    grid[y, x] = label
                else:
                    grid[y, x] = EMPTY

        exit_infos_raw = self.exit_detector.detect(frame)

        exits: List[Pos] = []
        exit_infos: List[ExitInfo] = []
        exit_types: Dict[Pos, str] = {}
        exit_opened: Dict[Pos, bool] = {}

        for e in exit_infos_raw:
            tiles = e.get("tiles", [e["tile"]])
            tiles = [(int(x), int(y)) for x, y in tiles]

            info = ExitInfo(
                tiles=tiles,
                direction=e.get("direction", "unknown"),
                exit_type=e.get("exit_type", "unknown"),
                opened=bool(e.get("opened", False)),
                score=float(e.get("score", 0.0)),
            )
            exit_infos.append(info)

            for x, y in tiles:
                if 0 <= x < ROOM_W and 0 <= y < ROOM_H:
                    p = (x, y)
                    if p not in exits:
                        exits.append(p)
                    
                    exit_types[p] = info.exit_type
                    exit_opened[p] = info.opened
                    grid[y, x] = EXIT

        # 2. 玩家覆盖修正
        player_info = self.player_detector.detect(frame)
        player = None
        player_px = None
        facing = "down"

        if player_info is not None:
            player = player_info["tile"]
            player_px = player_info.get("position_px", None)
            facing = player_info["facing"]
            px, py = player
            grid[py, px] = PLAYER

        # 3. 怪物覆盖修正
        monsters = []
        for m in self.monster_detector.detect_all(frame):
            tx, ty = m["tile"]
            monsters.append((tx, ty))
            grid[ty, tx] = MONSTER

        return self.grid_to_symbolic(
            grid, 
            player, 
            player_px,
            facing, 
            monsters, 
            exits, 
            exit_infos,
            exit_types,
            exit_opened
        )

    def grid_to_symbolic(self, grid, player, player_px, facing, monsters, exits_hint=None, exit_infos_hint=None, exit_types_hint=None, exit_opened_hint=None):
        chests = []
        traps = []
        buttons = []
        switches = []
        npcs = []
        gaps = []
        bridges = []
        exits = list(exits_hint) if exits_hint is not None else []
        exit_infos = list(exit_infos_hint) if exit_infos_hint is not None else []
        exit_types = dict(exit_types_hint) if exit_types_hint is not None else {}
        exit_opened = dict(exit_opened_hint) if exit_opened_hint is not None else {}

        for y in range(ROOM_H):
            for x in range(ROOM_W):
                v = int(grid[y, x])

                if v == CHEST:
                    chests.append((x, y))
                elif v == TRAP:
                    traps.append((x, y))
                elif v == BUTTON:
                    buttons.append((x, y))
                elif v == SWITCH:
                    switches.append((x, y))
                elif v == NPC:
                    npcs.append((x, y))
                elif v == GAP:
                    gaps.append((x, y))
                elif v == BRIDGE:
                    bridges.append((x, y))
                elif v == EXIT:
                    p = (x, y)
                    if p not in exits:
                        exits.append((x, y))
                    if p not in exit_types:
                        exit_types[p] = "unknown"
                    if p not in exit_opened:
                        exit_opened[p] = False  

        return SymbolicObs(
            grid=grid,
            player=player,
            player_px=player_px,
            facing=facing,
            monsters=monsters,
            chests=chests,
            exits=exits,
            exit_infos=exit_infos,
            exit_types=exit_types,
            exit_opened=exit_opened,
            traps=traps,
            buttons=buttons,
            switches=switches,
        )
    
@dataclass
class Subgoal:
    kind: str
    target: Optional[Pos] = None
    facing : Optional[int] = None

#lcd : 任务中可能发生的事件，复制到这里方便写代码
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
@dataclass
class Candidate:
    subgoal: Subgoal
    value: float
    dist: float
    risk: float
    score: float


class SymbolicPlanner:
    def next_subgoal(self, sym: SymbolicObs, belief: BeliefState) -> Subgoal:
        """
        上层 planner：决定现在应该干什么。
        generate a list of candidate subgoals, score them, and return the best one.
        """

        # 玩家位置识别失败时，不要乱动
        if sym.player is None:
            return Subgoal("wait")


        candidate_subgoals = []

        for chest in sym.chests:
            if chest in belief.opened_chests:
                continue
            candidate_subgoals.append(
                self.make_candidate(sym, belief, "find_chest", chest, self.base_value("find_chest", belief))
            )

        for monster in sym.monsters:
            if monster in belief.killed_monsters:
                continue
            candidate_subgoals.append(
                self.make_candidate(sym, belief, "attack_monster", monster, self.base_value("attack_monster", belief))
            )

        
        # 出口候选：按 ExitInfo 生成，一个双格出口只生成一个候选
        for info in sym.exit_infos:
            if not self.exit_is_usable(sym, belief, info):
                continue

            target_tile = nearest_tile(sym.player, info.tiles)
            if target_tile is None:
                continue

            candidate_subgoals.append(
                self.make_candidate(
                    sym,
                    belief,
                    "go_exit",
                    target_tile,
                    self.exit_value(sym, belief, info),
                )
            )
        
        for button in sym.buttons:
            if button in belief.pressed_buttons:
                continue
            candidate_subgoals.append(
                self.make_candidate(sym, belief, "press_button", button, self.base_value("press_button", belief))
            )
        
        for switch in sym.switches:
            candidate_subgoals.append(
                self.make_candidate(sym, belief, "activate_switch", switch, self.base_value("activate_switch", belief))
            )

        if candidate_subgoals:
            best = max(candidate_subgoals, key=lambda c: c.score)
            print("[PLAN_SELECT]", best.subgoal)
            return best.subgoal
        return Subgoal("explore")

    def nearest(self, start: Pos, candidates: List[Pos]) -> Optional[Pos]:
        if not candidates:
            return None
        sx, sy = start
        return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

    # score = 预期收益 - 路径代价 - 风险代价 - 时间代价
    def make_candidate(self, sym, belief, kind, target, value):
        assert sym.player is not None

        dist = self.estimate_distance(sym, kind, target)
        risk = self.estimate_risk(sym, target, belief)

        if kind == "attack_monster" :
            score = value - 0.5 * dist
        else:
            score = (
                value
                - 0.5 * dist
                - 2.0 * risk
                # - 0.05 * belief.no_progress_steps
            )

        return Candidate(
            subgoal=Subgoal(kind, target),
            value=value,
            dist=dist,
            risk=risk,
            score=score,
        )

    def base_value(self, kind, belief):
        if kind == "go_exit":
            return 20.0 

        if kind == "find_chest":
            return 15.0

        if kind == "attack_monster":
            return 18.0 if belief.has_sword else -999.0

        if kind == "press_button":
            return 12.0

        if kind == "activate_switch":
            return 10.0

        if kind == "explore":
            return 8.0

        return 0.0
    
    # def estimate_distance(self, sym, kind, target):
    #     if sym.player is None or target is None:
    #         return 999.0

    #     sx, sy = sym.player
    #     tx, ty = target
    #     return abs(sx - tx) + abs(sy - ty)
    def estimate_distance(self, sym, kind, target):
        if sym.player is None or target is None:
            return 999.0

        if kind == "attack_monster":
            best = 999.0

            for p in adjacent_tiles(target):
                if not in_bounds(p):
                    continue

                x, y = p
                if not is_passable(int(sym.grid[y, x])):
                    continue

                path = astar_path(sym.grid, sym.player, p, sym)

                if path or sym.player == p:
                    best = min(best, float(len(path)))

            return best

        path = astar_path(sym.grid, sym.player, target, sym)

        if not path and sym.player != target:
            return 999.0

        return float(len(path))
    
    def estimate_risk(self, sym, target, belief):
        if target is None:
            return 0.0

        tx, ty = target
        
        risk = 0.0

        for mx, my in sym.monsters:
            if abs(mx - tx) + abs(my - ty) <= 2:
                risk += 3.0
        
        for nx, ny in sym.traps:
            if abs(tx - nx) + abs(ty - ny) <= 1:
                risk += 2.0
        
        return risk
    
    def exit_is_usable(self, sym: SymbolicObs, belief: BeliefState, info: ExitInfo) -> bool:
        """
        判断这个出口当前值不值得尝试。
        注意：这是 planner 层判断，不是最终安全判断。
        """
        if info.opened:
            return True

        if info.exit_type == "normal":
            return True

        if info.exit_type == "locked_key":
            return belief.keys > 0

        if info.exit_type == "conditional":
            # 第一版：还有怪物时先不走条件门
            if sym.monsters:
                return False

            # 有按钮/开关时，先处理机关
            if sym.buttons or sym.switches or sym.chests:
                return False

            return True

        # unknown 的策略：
        # 有钥匙，或者当前没宝箱了，可以尝试。
        return belief.has_key or not sym.chests

    def exit_value(self, sym: SymbolicObs, belief: BeliefState, info: ExitInfo) -> float:
        """
        给出口候选一个基础价值。
        """
        if info.opened:
            return 80.0

        if info.exit_type == "locked_key":
            return 80.0 if belief.keys > 0 else -999.0

        if info.exit_type == "conditional":
            return 70.0

        if info.exit_type == "normal":
            # 普通门通常是换房间/探索入口，不一定是最终出口
            return 25.0

        return 20.0


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
    # return tile in {EMPTY, PLAYER, EXIT, BUTTON, BRIDGE, SWITCH}
    return tile in {EMPTY, PLAYER, BRIDGE} #暂时只能走空地、玩家、桥，其他都不走
    

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

def heuristic(a: Pos, b: Pos) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def tile_risk_cost(pos: Pos, sym: SymbolicObs) -> float:
    """
    额外风险成本：
    - 靠近怪物，加成本
    - 靠近陷阱，加成本
    第一版先简单写，后面可以继续调。
    """
    x, y = pos
    cost = 0.0

    for mx, my in sym.monsters:
        d = abs(mx - x) + abs(my - y)
        if d == 0:
            cost += 100.0
        elif d == 1:
            cost += 8.0
        elif d == 2:
            cost += 3.0

    for tx, ty in sym.traps:
        d = abs(tx - x) + abs(ty - y)
        if d == 0:
            cost += 100.0
        elif d == 1:
            cost += 5.0

    return cost


def astar_path(
    grid: np.ndarray,
    start: Pos,
    goal: Pos,
    sym: Optional[SymbolicObs] = None,
) -> List[int]:
    """
    返回 tile 级动作序列。
    如果 sym 不为空，会加入风险成本，尽量绕开怪物/陷阱附近。
    """
    if start == goal:
        return []

    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    parent: Dict[Pos, Tuple[Optional[Pos], Optional[int]]] = {
        start: (None, None)
    }

    g_score: Dict[Pos, float] = {
        start: 0.0
    }

    while open_heap:
        _, cur_g, cur = heapq.heappop(open_heap)

        if cur == goal:
            actions = []
            p = cur
            while parent[p][0] is not None:
                prev, act = parent[p]
                actions.append(act)
                p = prev
            actions.reverse()
            return actions

        # 如果这是旧的 heap entry，跳过
        if cur_g > g_score.get(cur, float("inf")):
            continue

        for nxt, act in neighbors(cur):
            if not in_bounds(nxt):
                continue

            x, y = nxt
            tile = int(grid[y, x])

            if not is_passable(tile):
                continue

            step_cost = 1.0

            if sym is not None:
                step_cost += tile_risk_cost(nxt, sym)

            new_g = g_score[cur] + step_cost

            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                parent[nxt] = (cur, act)

                f = new_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, new_g, nxt))

    return []


def repeat_action(action: int, n: int) -> List[int]:
    return [action] * n


def expand_tile_actions(tile_actions: List[int]) -> List[int]:
    """将tile级别移动转化为pixel也就是像素级别pixel_actions"""
    pixel_actions = []
    for a in tile_actions:
        pixel_actions.extend(repeat_action(a, MOVE_REPEAT))
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

# def is_encounter_monster(sym : SymbolicObs,bound = 10) -> int | None :
#     """
#     判断是否遭遇monster，如果是，返回monster所在方向facing,如果否，返回ACTION_NOOP=0
#     遭遇是指player与monster距离近(bound)，可以直接朝facing方向进行攻击
#     """
#     player_px = sym.player_px
#     monster_px = nearest_px(player_px,sym.monsters_px)

#     if monster_px is None:
#         return ACTION_NOOP
#     #monster位于right
#     if (monster_px[0] < player_px[0] + TILE_SIZE + bound) and (monster_px[0] > player_px[0] + TILE_SIZE) and \
#             (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
#         print(monster_px[0] , player_px[0] + TILE_SIZE + bound)

#         return ACTION_RIGHT

#     #monster位于left
#     if (monster_px[0] + TILE_SIZE > player_px[0] - bound) and (monster_px[0] + TILE_SIZE < player_px[0]) and \
#             (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
#         return ACTION_LEFT

#     #monster位于up
#     if (monster_px[1] + TILE_SIZE > player_px[1] - bound) and (monster_px[1] + TILE_SIZE < player_px[1]) and \
#             (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
#         return ACTION_UP

#     #monster位于down
#     if (monster_px[1] < player_px[1] + TILE_SIZE+ bound) and (monster_px[1] > player_px[1] + TILE_SIZE) and\
#             (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
#         return ACTION_DOWN

#     return ACTION_NOOP


# class SymbolicPlanner:
#     def next_subgoal(self, sym: SymbolicObs, belief: BeliefState) -> Subgoal:
#         """
#         上层 planner：决定现在应该干什么。
#         先实现 Task 1/2/3通用逻辑：
#         1. detect_near_monster -> hit_monster
#         1. detect_chest_unopened -> find_chest
#         2. have_key_and_detect_closedExit -> openExit_leave
#         3. detect_normal_opened_exit -> leave
#         """


#         # 玩家位置识别失败时，不要乱动
#         if sym.player is None:
#             return Subgoal("wait")

#         # 附近有monster
#         monster_facing = is_encounter_monster(sym)
#         if monster_facing:
#             return Subgoal("kill_monster",facing=monster_facing)

#         # 没钥匙：优先去最近宝箱 detect_chest_unopened -> open_chest
#         # if not belief.has_key:
#         #     chest = self.nearest(sym.player, sym.chests)
#         #     if chest is not None:
#         #         return Subgoal("find_chest", chest)
#         #     return Subgoal("explore")
#         chest = self.nearest(sym.player,sym.chests)
#         if chest is not None:
#             return Subgoal("find_chest",chest)

#         # 有钥匙：去出口
#         if belief.has_key:
#             exit_pos = self.nearest(sym.player, sym.exits)
#             if exit_pos is not None:
#                 return Subgoal("go_exit", exit_pos)

#         return Subgoal("explore")

#     def nearest(self, start: Pos, candidates: List[Pos]) -> Optional[Pos]:
#         """从candidates中找到距离start最近的一个"""
#         if not candidates:
#             return None
#         sx, sy = start
#         return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

def action_to_name(action: int) -> str:
    """将action(int)转换为str"""
    if action == ACTION_UP:
        return "up"
    if action == ACTION_DOWN:
        return "down"
    if action == ACTION_LEFT:
        return "left"
    if action == ACTION_RIGHT:
        return "right"
    return "none"


def manhattan(a: Pos, b: Pos) -> int:
    """计算两个tile之间的曼哈顿距离"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_exit_info_for_tile(sym: SymbolicObs, tile: Pos) -> Optional[ExitInfo]:
    """根据 tile 返回对应的 ExitInfo。"""
    for info in sym.exit_infos:
        if tile in info.tiles:
            return info
    return None

def exit_out_action(info: ExitInfo) -> int:
    """根据出口信息生成向外移动的动作"""
    if info.direction == "north":
        return ACTION_UP
    if info.direction == "south":
        return ACTION_DOWN
    if info.direction == "west":
        return ACTION_LEFT
    if info.direction == "east":
        return ACTION_RIGHT
    return ACTION_NOOP


def exit_approach_tiles(info: ExitInfo) -> List[Pos]:
    """
    返回门内侧可站的格子。
    west 门 tiles=[(0,3),(0,4)]，内侧是 [(1,3),(1,4)]
    north 门 tiles=[(4,0),(5,0)]，内侧是 [(4,1),(5,1)]
    """
    result = []

    for x, y in info.tiles:
        if info.direction == "north":
            p = (x, y + 1)
        elif info.direction == "south":
            p = (x, y - 1)
        elif info.direction == "west":
            p = (x + 1, y)
        elif info.direction == "east":
            p = (x - 1, y)
        else:
            continue

        if in_bounds(p):
            result.append(p)

    # 去重
    out = []
    for p in result:
        if p not in out:
            out.append(p)
    return out


def nearest_tile(start: Pos, tiles: List[Pos]) -> Optional[Pos]:
    """从 tiles 中找到距离 start 最近的 tile。"""
    if not tiles:
        return None

    sx, sy = start
    return min(tiles, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))


def align_action_before_move(sym: SymbolicObs, next_action: int, tolerance: float = 1.5) -> int:
    """
    如果下一步要竖直移动，先让 x 对齐当前 tile。
    如果下一步要水平移动，先让 y 对齐当前 tile。
    避免贴边转弯撞墙。
    """
    if sym.player is None or sym.player_px is None:
        return ACTION_NOOP

    tx, ty = sym.player
    px, py = sym.player_px

    target_px = tx * TILE_SIZE
    target_py = ty * TILE_SIZE

    # 要上下走，先校正 x
    if next_action in {ACTION_UP, ACTION_DOWN}:
        dx = px - target_px

        if dx < -tolerance:
            return ACTION_RIGHT
        if dx > tolerance:
            return ACTION_LEFT

    # 要左右走，先校正 y
    if next_action in {ACTION_LEFT, ACTION_RIGHT}:
        dy = py - target_py

        if dy < -tolerance:
            return ACTION_DOWN
        if dy > tolerance:
            return ACTION_UP

    return ACTION_NOOP

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
            return self.actions_to_exit(sym, belief, subgoal.target)

        if subgoal.kind == "attack_monster" and subgoal.target is not None:
            return self.actions_attack_monster(sym, belief, subgoal.target)
        
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

    
    def actions_to_exit(self, sym: SymbolicObs, belief: BeliefState, exit_pos: Pos) -> List[int]:
        """先走到出口 tile，然后再朝门外走(一步一规划)"""
        assert sym.player is not None

        info = get_exit_info_for_tile(sym, exit_pos)

        # 有 ExitInfo：先走到门内侧，再持续朝门外走
        if info is not None:
            out_action = exit_out_action(info)
            approach_candidates = []

            for p in exit_approach_tiles(info):
                x, y = p

                # 门内侧必须可走
                if not is_passable(int(sym.grid[y, x])):
                    continue

                # 不要踩回刚打开的宝箱位置；opened chest 往往仍可能阻挡
                if p in belief.opened_chests:
                    continue

                # path = astar_path(sym.grid, sym.player, p, sym)
                path = bfs_path(sym.grid, sym.player, p)
                if path or sym.player == p:
                    approach_candidates.append((len(path), p, path))

            if approach_candidates:
                approach_candidates.sort(key=lambda t: t[0])
                _, approach, tile_actions = approach_candidates[0]

                if sym.player == approach:
                    # 玩家已经在门内侧了，直接朝外走
                    return [out_action] * 64
                
                if tile_actions:
                    first = tile_actions[0]
                    align_action = align_action_before_move(sym, first, tolerance=0.5)
                    actions = []

                    if align_action != ACTION_NOOP:
                        actions.extend([align_action] * 3)

                    actions.extend([first] * MOVE_REPEAT)
                    print(
                    "[EXIT_STEP_PLAN]",
                    "player=", sym.player,
                    "player_px=", sym.player_px,
                    "approach=", approach,
                    "full_path=", tile_actions,
                    "first=", first,
                    "align=", align_action,
                    "queue=", len(actions),
                    )

                return actions

            print(
                "[EXIT_PLAN_FAIL]",
                "exit_tiles=", info.tiles,
                "dir=", info.direction,
                "approach_tiles=", exit_approach_tiles(info),
                "player=", sym.player,
            )

        # fallback：没有 ExitInfo，旧逻辑
        tile_actions = bfs_path(sym.grid, sym.player, exit_pos)
        actions = expand_tile_actions(tile_actions)

        out_action = self.exit_direction_from_tile(exit_pos)
        if out_action != ACTION_NOOP:
            actions.extend([out_action] * 64)

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

        # tile_actions = bfs_path(sym.grid, sym.player, target_adj)
        tile_actions = astar_path(sym.grid, sym.player, target_adj, sym)
        actions = expand_tile_actions(tile_actions)

        # 到达相邻格后，如果角色朝向不对，移动一步方向键让角色朝向宝箱
        face_action = action_to_face(target_adj, obj_pos)
        if face_action != ACTION_NOOP and face_action != sym.facing:
            actions.append(face_action)

        # 按 A
        actions.append(ACTION_A)
        return actions
    
    def actions_attack_monster(
    self,
    sym: SymbolicObs,
    belief: BeliefState,
    monster_pos: Pos
    ) -> List[int]:
        """
        杀怪 option:
        1. 如果已经相邻，面向怪物，然后按 A 多次。
        2. 如果不相邻，走到一个相邻格。
        3. 优先选择“最后一步移动方向正好面向怪物”的相邻格，这样不用额外撞怪物来转向。
        """
        assert sym.player is not None

        player = sym.player
        mx, my = monster_pos

        # 已经相邻：直接攻击
        if manhattan(player, monster_pos) == 1:
            face_action = action_to_face(player, monster_pos)
            desired_facing = action_to_name(face_action)

            actions = []

            # 如果当前 belief 方向不是朝怪物，先尝试转向
            # 注意：这个动作可能会被 shield 拦，后面 Policy 里会专门放行一次 face_monster
            if belief.facing != desired_facing and face_action != ACTION_NOOP:
                actions.append(face_action)

            # 多按几次 A，防止一次没打死
            actions.extend([ACTION_A] * 10)

            print(
                "[ATTACK_PLAN]",
                "already_adjacent",
                "player=", player,
                "monster=", monster_pos,
                "facing=", belief.facing,
                "need=", desired_facing,
                "actions=", actions,
            )

            return actions

        # 不相邻：找怪物周围可站的位置
        candidates = []

        for p in adjacent_tiles(monster_pos):
            if not in_bounds(p):
                continue

            x, y = p
            if not is_passable(int(sym.grid[y, x])):
                continue

            path = bfs_path(sym.grid, player, p)

            # 不可达跳过
            if not path and player != p:
                continue

            face_action = action_to_face(p, monster_pos)

            # 重点：优先选最后一步移动方向 == 面向怪物方向的点
            # 例如站在怪物左边，最后一步最好是 RIGHT，这样到位后天然面向怪物。
            orientation_penalty = 0
            if path:
                if path[-1] != face_action:
                    orientation_penalty = 8
            else:
                orientation_penalty = 0

            score = len(path) + orientation_penalty

            candidates.append((score, p, path, face_action))

        if not candidates:
            print("[ATTACK_PLAN_FAIL]", "monster=", monster_pos, "no adjacent candidate")
            return [ACTION_NOOP]

        candidates.sort(key=lambda t: t[0])
        score, target_adj, tile_actions, face_action = candidates[0]

        actions = expand_tile_actions(tile_actions)

        # 如果最后一步没有自然面向怪物，就补一个转向动作
        if not tile_actions or tile_actions[-1] != face_action:
            if face_action != ACTION_NOOP:
                actions.append(face_action)

        actions.extend([ACTION_A] * 10)

        print(
            "[ATTACK_PLAN]",
            "player=", player,
            "monster=", monster_pos,
            "target_adj=", target_adj,
            "tile_actions=", tile_actions,
            "face_action=", face_action,
            "score=", score,
            "queue=", len(actions),
        )

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
            self.belief.step += 1
            self.belief.update_facing_from_action(self.force_exit_action)
            self.belief.last_action = self.force_exit_action

            if self.force_exit_steps % 10 == 0:
                print(
                    "[FORCE_EXIT]",
                    "step=", self.belief.step,
                    "action=", self.force_exit_action,
                    "steps_left=", self.force_exit_steps,
                )

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
            self.handle_debug_events(info)
        else:
            sym = self.last_sym
            self.belief.step += 1

        replanned = False
        # # 附近有monster
        # encounter_monster = is_encounter_monster(sym,)
        # if encounter_monster:
        #     replanned = True

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

        # attack_monster 时，允许一次“朝怪物方向”的移动动作通过，
        # 主要目的是让角色转向怪物。
        if self.is_face_monster_action(sym, raw_action):
            print(
                "[FACE_MONSTER]",
                "step=", self.belief.step,
                "player=", sym.player,
                "monster=", self.current_subgoal.target,
                "action=", raw_action,
            )

            self.belief.update_facing_from_action(raw_action)
            self.belief.last_action = raw_action
            return int(raw_action)
        
        required_exit_action = self.exit_action_if_on_current_exit(sym)


        if (
            self.current_subgoal is not None
            and self.current_subgoal.kind == "go_exit"
            and self.is_border_leaving_action(sym.player, raw_action)
            and raw_action == required_exit_action
        ):
            self.force_exit_action = raw_action
            self.force_exit_steps = 40

            # 防止 force_exit 结束后继续吃旧队列，又反复 START_FORCE_EXIT
            self.action_queue.clear()
            print(
                "[START_FORCE_EXIT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "raw=", raw_action,
                "exit_target=", self.current_subgoal.target,
                "exit_action=", required_exit_action,
                "exits=", sym.exits,
                "exit_infos=",
                [
                    {
                        "tiles": info.tiles,
                        "dir": info.direction,
                        "type": info.exit_type,
                        "opened": info.opened,
                    }
                    for info in sym.exit_infos
                ],
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
                # "monster_px=",sym.monsters_px,
                "raw=", raw_action,
                "safe=", action,
                "queue_left=", len(self.action_queue),
                "stuck=", self.belief.stuck_count,
                "subgoal=", self.current_subgoal,
            )

        self.belief.update_facing_from_action(action)
        self.belief.last_action = action
        return int(action)

    def need_replan(self, sym: SymbolicObs, info=None,force_replan=False) -> bool:
        """判断是否需要重新规划"""
        #lcd : 强制replan,发生特定事情需要，比如有monster靠近
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
    
    def is_face_monster_action(self, sym: SymbolicObs, action: int) -> bool:
        if sym.player is None:
            return False

        if self.current_subgoal is None:
            return False

        if self.current_subgoal.kind != "attack_monster":
            return False

        monster = self.current_subgoal.target
        if monster is None:
            return False

        # 必须已经相邻
        if manhattan(sym.player, monster) != 1:
            return False

        # 当前动作必须正好是朝怪物方向
        return action_to_face(sym.player, monster) == action
    
    def exit_action_if_on_current_exit(self, sym: SymbolicObs) -> Optional[int]:
        """
        只有玩家真的站在当前目标出口的两个 tile 之一，才返回出门方向。
        不能只看 x==0/y==0，否则会在门旁边的墙边误触发 force_exit。
        """
        if sym.player is None:
            return None

        if self.current_subgoal is None:
            return None

        if self.current_subgoal.kind != "go_exit":
            return None

        target = self.current_subgoal.target
        if target is None:
            return None

        info = get_exit_info_for_tile(sym, target)

        # 如果有完整 ExitInfo，用双格出口判断
        if info is not None:
            px, py = sym.player
            xs = [x for x, y in info.tiles]
            ys = [y for x, y in info.tiles]

            if info.direction == "north":
                if py == 0 and min(xs) <= px <= max(xs):
                    return ACTION_UP

            if info.direction == "south":
                if py == ROOM_H - 1 and min(xs) <= px <= max(xs):
                    return ACTION_DOWN

            if info.direction == "west":
                if px == 0 and min(ys) <= py <= max(ys):
                    return ACTION_LEFT

            if info.direction == "east":
                if px == ROOM_W - 1 and min(ys) <= py <= max(ys):
                    return ACTION_RIGHT

            return None

        # fallback：没有 ExitInfo 时，只允许站在 target 本格出门
        if sym.player == target:
            x, y = target
            if y == 0:
                return ACTION_UP
            if y == ROOM_H - 1:
                return ACTION_DOWN
            if x == 0:
                return ACTION_LEFT
            if x == ROOM_W - 1:
                return ACTION_RIGHT

        return None
    
    def handle_debug_events(self, info=None):
        if not isinstance(info, dict):
            return

        events = info.get("events", {})
        if not isinstance(events, dict):
            return

        flags = events.get("flags", {}) or {}

        if flags.get("chest_opened", False):
            if (
                self.current_subgoal is not None
                and self.current_subgoal.kind == "find_chest"
                and self.current_subgoal.target is not None
            ):
                self.belief.opened_chests.add(self.current_subgoal.target)
                print("[MEMORY] opened_chest=", self.current_subgoal.target)
    
def make_policy() -> Policy:
    return Policy()