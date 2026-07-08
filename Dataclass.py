from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
import numpy as np

import heapq

# lcd : 将这些枚举从其他文件中导入，不重复定义
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
TILE = 16
TILE_SIZE = 16

EMPTY = 0
WALL = 1
PLAYER = 2
MONSTER = 3
CHEST = 4
EXIT = 5
TRAP = 6
BUTTON = 7
NPC = 8
GAP = 9
BRIDGE = 10
SWITCH = 11
CHEST_OPENED = 12
UNKNOWN = 13

# grid : [8 , 10]
ROOM_W = 10
ROOM_H = 8

# 数据结构类型
Pos = Tuple[int, int]
pxPos = Tuple[float, float]  # 像素位置


# 识别exit
@dataclass
class ExitInfo:
    tiles: List[Pos]  # 这个出口占据的两个 tile
    direction: str  # north/south/west/east
    exit_type: str = "unknown"  # normal/locked_key/conditional/unknown
    opened: bool = False
    score: float = 0.0

    # 联通信息
    dest: int = 0  # 目标房间id
    start: int = 0  # 所在房间id

    is_reached: bool = False  # 是否到达过（指到达到dest）

    @property
    def representative(self) -> Pos:
        return self.tiles[0]


@dataclass
class SymbolicObs:
    grid: np.ndarray  # shape: (8, 10)
    player: Optional[Pos] = None
    facing: str = "up"
    monsters: List[Pos] = field(default_factory=list)
    chests: List[Pos] = field(default_factory=list)
    exits: List[Pos] = field(default_factory=list)
    traps: List[Pos] = field(default_factory=list)
    buttons: List[Pos] = field(default_factory=list)
    switches: List[Pos] = field(default_factory=list)

    # lcd : 添加player与monster的具体像素坐标
    player_px: Optional[pxPos] = None
    monsters_px: List[pxPos] = field(default_factory=list)

    # lcd : 添加exits的信息
    exit_infos: dict[str, Optional[ExitInfo]] = field(default_factory=dict)  # 记录东西南北4个门的类型，状态，是否存在（不存在为None）

    # exit_types: Dict[Pos, str] = field(default_factory=dict)
    # exit_opened: Dict[Pos, bool] = field(default_factory=dict)

    @property
    def exit_types(self) -> Dict[Pos, str]:
        """动态构建 tile -> exit_type 的映射"""
        result = {}
        for info in self.exit_infos.values():
            if info is not None:
                for tile in info.tiles:
                    result[tile] = info.exit_type
        return result

    @property
    def exit_opened(self) -> Dict[Pos, bool]:
        """动态构建 tile -> opened 的映射"""
        result = {}
        for info in self.exit_infos.values():
            if info is not None:
                for tile in info.tiles:
                    result[tile] = info.opened
        return result


@dataclass
class BeliefState:
    task_id: Optional[str] = None
    step: int = 0

    # 任务进度
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
        # 任务进度
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
            self.has_sword = ("sword" in self.tools) or ("sword" in self.items) or (
                    inv.get("equipped", {}).get("A") == "sword")

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
    facing: Optional[int] = None
    dest_room_id: Optional[int] = None
    start_room_id: Optional[int] = None
    exit_dir: Optional[str] = None

@dataclass
class Candidate:
    subgoal: Subgoal
    value: float
    dist: float
    risk: float
    score: float

