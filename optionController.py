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

from Dataclass import  BeliefState,SymbolicObs,Subgoal

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

def opposition(dir: str) -> str | None:
    """返回相反方向"""
    if dir == "north":
        return 'down'
    if dir == "south":
        return 'up'
    if dir == "west":
        return 'right'
    if dir == "east":
        return 'left'
    else:
        # 不合法返回None
        return None


def neighbors(p: Pos) -> List[Tuple[Pos, int]]:
    x, y = p
    return [
        ((x, y - 1), ACTION_UP),
        ((x, y + 1), ACTION_DOWN),
        ((x - 1, y), ACTION_LEFT),
        ((x + 1, y), ACTION_RIGHT),
    ]

# lcd
def nearest_px(start: pxPos, candidates: List[pxPos]) -> Optional[pxPos]:
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


def is_monster(tile: int) -> bool:
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

            # lcd : 无视monster
            if (not is_passable(int(grid[y, x]))) and (not is_monster(int(grid[y, x]))):
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


def Str2Enum_facing(facing: str) -> int:
    """建立方向facing(str)与action(int)之间的转换"""
    relation = {
        "up": ACTION_UP,
        "down": ACTION_DOWN,
        "left": ACTION_LEFT,
        "right": ACTION_RIGHT
    }
    return relation.get(facing, ACTION_NOOP)


def Enum2Str_facint(facing: int) -> str:
    relation = {
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right"
    }
    return relation.get(facing, 'wait')


# lcd
def is_encounter_monster(sym: SymbolicObs, bound=10) -> int | None:
    """
    判断是否遭遇monster，如果是，返回monster所在方向facing,如果否，返回ACTION_NOOP=0
    遭遇是指player与monster距离近(bound)，可以直接朝facing方向进行攻击
    """
    player_px = sym.player_px
    monster_px = nearest_px(player_px, sym.monsters_px)

    if monster_px is None:
        return ACTION_NOOP
    # monster位于right
    if (monster_px[0] < player_px[0] + TILE_SIZE + bound) and (monster_px[0] > player_px[0] + TILE_SIZE) and \
            (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
        print(monster_px[0], player_px[0] + TILE_SIZE + bound)

        return ACTION_RIGHT

    # monster位于left
    if (monster_px[0] + TILE_SIZE > player_px[0] - bound) and (monster_px[0] + TILE_SIZE < player_px[0]) and \
            (monster_px[1] > player_px[1] - TILE_SIZE) and (monster_px[1] < player_px[1] + TILE_SIZE):
        return ACTION_LEFT

    # monster位于up
    if (monster_px[1] + TILE_SIZE > player_px[1] - bound) and (monster_px[1] + TILE_SIZE < player_px[1]) and \
            (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
        return ACTION_UP

    # monster位于down
    if (monster_px[1] < player_px[1] + TILE_SIZE + bound) and (monster_px[1] > player_px[1] + TILE_SIZE) and \
            (monster_px[0] > player_px[0] - TILE_SIZE) and (monster_px[0] < player_px[0] + TILE_SIZE):
        return ACTION_DOWN

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

        if subgoal.kind == "find_monster" and subgoal.target is not None:
            return self.actions_to_interactable(sym, subgoal.target)

        if subgoal.kind == "go_exit" and subgoal.target is not None:
            return self.actions_to_exit(sym, subgoal.target)

        if subgoal.kind == 'switch' and subgoal.target is not None:
            return self.actions_to_interactable(sym, subgoal.target)

        if subgoal.kind == "explore":
            # 最简单探索：先等一下，后面再做 frontier exploration
            return [ACTION_NOOP]

        return [ACTION_NOOP]

    def actions_to_fit_tile(self, sym: SymbolicObs, dim: int = 0) -> List[int]:
        """为了避免'action_blocked'，可以在bfs前将player移动到更好契合tile,dim表示在那个维度进行对齐，0-x,1-y,2-xy"""
        actions = []
        px_x, px_y = sym.player_px
        t_px_x, t_px_y = sym.player
        t_px_x = t_px_x * TILE_SIZE
        t_px_y = t_px_y * TILE_SIZE

        action_x, action_y = ACTION_LEFT, ACTION_UP
        if px_x < t_px_x:
            action_x = ACTION_RIGHT
        if px_y < t_px_y:
            action_y = ACTION_DOWN

        if dim == 2:
            return [action_x] * abs(px_x - t_px_x) + [action_y] * abs(px_y - t_px_y)
        if dim == 0:
            return [action_x] * abs(px_x - t_px_x)
        if dim == 1:
            return [action_y] * abs(px_y - t_px_y)
        else:
            return []

    def actions_to_kill_monster(self, sym: SymbolicObs, facing: int) -> List[int]:
        """
        #lcd
        朝指定方向进行攻击
        面向monster并攻击
        """
        assert sym.player is not None
        actions = []
        # lcd : 如果player的朝向已经正确，pass
        if facing != ACTION_NOOP and facing != Str2Enum_facing(sym.facing):
            actions.append(facing)
        # 按 A
        actions.append(ACTION_A)
        return actions

    def actions_to_exit(self, sym: SymbolicObs, exit_pos: Pos) -> List[int]:
        """获取前往exit的actions"""
        assert sym.player is not None

        actions = []
        # 先契合tile
        actions += self.actions_to_fit_tile(sym)

        # 先走到出口 tile
        tile_actions = bfs_path(sym.grid, sym.player, exit_pos)
        actions += expand_tile_actions(tile_actions)

        # 再朝边界方向多走 1 步
        out_action = self.exit_direction_from_tile(exit_pos)
        print(f'out_action:{Enum2Str_facint(out_action)}')
        if out_action != ACTION_NOOP:
            actions.extend([out_action] * 2)
            print(actions)
        print(actions)
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
        # lcd : 如果player的朝向已经正确，pass
        if face_action != ACTION_NOOP and face_action != sym.facing:
            actions.append(face_action)

        # 按 A
        actions.append(ACTION_A)
        return actions