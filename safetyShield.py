from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
import numpy as np

import heapq
# lcd : 将能够放到vision_exact.py的代码尽量放到通过导入来使用，不再重复定义
from vision_exact import Pos, pxPos, SymbolicObs, PixelPerception, ExitInfo

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

# 符号 tile 编码，先和文档里的 grid code 保持一致
from Dataclass import (
    EMPTY,
    WALL,
    PLAYER,
    MONSTER,
    CHEST,
    EXIT,
    TRAP,
    BUTTON,
    NPC,
    GAP,
    BRIDGE,
    SWITCH,
    # gird's metadata
    TILE_SIZE,
    ROOM_W,
    ROOM_H,
)

from Dataclass import  BeliefState,SymbolicObs

def in_bounds(p: Pos) -> bool:
    """判断是否在grid合法范围内"""
    x, y = p
    return 0 <= x < ROOM_W and 0 <= y < ROOM_H

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

            # 不主动走进墙、陷阱、gap
            if tile in {WALL, TRAP, GAP}:
                return ACTION_NOOP
            # 如果发现怪兽
            # if tile == MONSTER:
            #     return ACTION_A

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
            # print(f'not exit--{action}')
            return False

        return (
                (y == 0 and action == ACTION_UP) or
                (y == ROOM_H - 1 and action == ACTION_DOWN) or
                (x == 0 and action == ACTION_LEFT) or
                (x == ROOM_W - 1 and action == ACTION_RIGHT)
        )