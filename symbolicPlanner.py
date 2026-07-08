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

from Dataclass import  BeliefState,SymbolicObs,Subgoal,Candidate

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

def adjacent_tiles(pos: Pos) -> List[Pos]:
    """return tiles : List[Pos] adjacent to pos : Pos"""
    x, y = pos
    return [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]

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


class SymbolicPlanner:
    def __init__(self):
        self.current_subgoal: Optional[Subgoal] = None
        # 房间 metadata
        self.rooms = {"explored": [], "unexplored": [0], "stillNeed": []}  # 管理已发现或潜在的房间数,初始时0房间还没探索,+管理需要二次进入的房间
        self.stillNeedIdx = 0  # 指向该访问的stillNeed索引
        self.room_num = 1  # 房间总数,默认出生房room_id=0

        self.current_room_id = 0
        self.last_room_id = 0

        self.room_ID2Coord = {0: (0, 0)}  # 房间坐标，以出生房间0为原点
        self.room_Coord2ID = {(0, 0): 0}  # 当前房间坐标到房间id的映射

        # 管理房间exit
        self.room_exits_info: Dict[int, Dict[str, ExitInfo | None]] = {
            0: {"north": None, "south": None, "west": None, "east": None}}  # 以房间room_id为索引

        self.has_key = False
        self.switched = False  # 每次离开switch房间后置零

    def next_subgoal(self, sym: SymbolicObs, belief: BeliefState) -> Subgoal:
        """
        上层 planner：决定现在应该干什么。
        先实现 Task 1/2/3通用逻辑：
        1. detect_near_monster -> hit_monster
        2. detect_chest_unopened -> find_chest
        3. have_key_and_detect_closedExit -> openExit_leave
        4. detect_normal_opened_exit -> leave
        5.explored_all_room_and_detect_switch -> activate_switch
        """

        self.has_key = belief.has_key
        # 玩家位置识别失败时，不要乱动
        if sym.player is None:
            self.current_subgoal = Subgoal('wait')
            return self.current_subgoal

        # lcd
        # 1.附近有monster
        monster_facing = is_encounter_monster(sym)
        if monster_facing:
            self.current_subgoal = Subgoal('kill_monster', facing=monster_facing)
            return Subgoal("kill_monster", facing=monster_facing)

        # lcd
        # 2.发现未打开chest
        chest = self.nearest(sym.player, sym.chests)
        if chest is not None:
            self.current_subgoal = Subgoal('find_chest', chest)
            return Subgoal("find_chest", chest)

        #2.5 房间内有monster
        monster = self.nearest(sym.player, sym.monsters)
        if monster is not None:
            self.current_subgoal = Subgoal('find_monster', monster)
            return Subgoal('find_monster',monster)

        # lcd
        # 3. 当前未将所有房间探索完毕且发现exit -> leave
        if self.rooms['unexplored']:
            for room_togo in reversed(self.rooms['unexplored']):  # 逆序找一个可以探索的且未探索的房间，dfs
                # 根据当前房间坐标和目标房间坐标找一条路径

                dir = self._bfs(self.current_room_id, room_togo)
                print(f'roomtogo{room_togo}, {dir}')
                if dir is not None and (dir != 'wait'):
                    exit = self.room_exits_info[self.current_room_id][dir]
                    if exit:
                        exit_pos = self.nearest(sym.player, exit.tiles)
                        print(f'exit_pos:{exit_pos}')
                        if exit_pos is not None:
                            self.current_subgoal = Subgoal("go_exit", exit_pos,
                                                           start_room_id=self.current_room_id,
                                                           dest_room_id=exit.dest,
                                                           exit_dir=dir)
                            return self.current_subgoal

        # lcd
        # 4. 当前所有房间探索完毕且需要重复访问的房间stillNeed不为空 -> 回去的exit and leave
        if self.rooms['stillNeed']:
            # 根据当前房间坐标和目标房间坐标找一条路径
            dir = self._bfs(self.current_room_id, self.rooms['stillNeed'][self.stillNeedIdx])
            if dir is not None:
                exit = self.room_exits_info[self.current_room_id][dir]
                if exit:
                    exit_pos = self.nearest(sym.player, exit.tiles)
                    print(f'exit_pos:{exit_pos}')
                    if exit_pos is not None:
                        self.current_subgoal = Subgoal("go_exit", exit_pos,
                                                       start_room_id=self.current_room_id, dest_room_id=exit.dest,
                                                       exit_dir=dir)
                        return self.current_subgoal

        # 5. switch
        switch = self.nearest(sym.player, sym.switches)
        if switch is not None:
            self.current_subgoal = Subgoal('switch', switch)
            return Subgoal("switch", target=switch)

        return Subgoal("explore")

    def make_candidate(self, sym, belief, kind, target, value):
        assert sym.player is not None

        dist = self.estimate_distance(sym, kind, target)
        risk = self.estimate_risk(sym, target, belief)

        if kind == "attack_monster":
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

    def neighbors(self, room_coord: Pos) -> List[Tuple[Pos, str]]:
        """返回与rood_coord相邻的房间坐标，并附加目标方位"""
        x, y = room_coord
        return [
            ((x, y - 1), 'north'),
            ((x, y + 1), 'south'),
            ((x - 1, y), 'west'),
            ((x + 1, y), 'east'),
        ]

    def explore_room(self, sym: SymbolicObs):
        """
        #lcd
        对id为room_id的新房间进行初步探索，更新房间 metadata
        self.current_room_id已经是新房间id
        self.last_room_id为来时的旧房间id
        """
        if self.current_room_id not in self.rooms['unexplored']:
            # 合法性检测
            return
        self.rooms['explored'].append(self.current_room_id)
        self.rooms['unexplored'].remove(self.current_room_id)
        # 还需要判断房间是否特殊，需要多次访问
        stillNeed = (sym.switches is not None) and (len(sym.switches) > 0)

        # 根据存在的exit，更新self.rooms['unexplored']，将exit里面当作潜在房间
        # 一般四个方向每个方向最多有一个exit
        dirs = ["north", "south", "west", "east"]
        exit_infos = sym.exit_infos
        print('debuging<<<<<<<<<<<<<<<<<<<<')
        print(exit_infos)
        for dir in dirs:
            if exit_infos.get(dir, None) is None:
                # 确定这个方向的exit存在
                print('continue')
                continue
            if ((opposition(dir) == sym.facing) and
                    (self.current_subgoal is not None) and (self.current_subgoal.kind == 'go_exit') and
                    (self.current_room_id != 0)):
                # 来时exit,0房间没有
                print('come on')

                self.room_exits_info[self.current_room_id][dir] = ExitInfo(
                    tiles=exit_infos[dir].tiles,
                    exit_type=exit_infos[dir].exit_type,
                    opened=exit_infos[dir].opened,
                    dest=self.last_room_id,  # 通往的房间id
                    start=self.current_room_id,
                    is_reached=True,  # 是否已经到达过dest房间
                    direction=dir,
                )
                if self.current_room_id == 2:
                    print(self.room_exits_info[self.current_room_id][dir])
            else:
                # 其他方向的exit
                # 每个exit都通往潜在的房间
                self.init_new_room(dir, sym)

        # 如果判断房间还需要
        if stillNeed:
            self.rooms['stillNeed'].append(self.current_room_id)

    def init_new_room(self, dir, sym: SymbolicObs):
        """初始化一个新房间"""
        # 其他方向的exit
        # 每个exit都通往潜在的房间
        print('init new room-', self.room_num)
        exit_infos = sym.exit_infos
        if exit_infos.get(dir, None) is None:
            return
        new_room_id = self.room_num
        self.room_num += 1
        self.rooms['unexplored'].append(new_room_id)
        # 计算新房间坐标
        current_room_coord = self.room_ID2Coord[self.current_room_id]
        new_x = current_room_coord[0]
        new_y = current_room_coord[1]
        if dir == 'north':
            new_y -= 1
        if dir == 'south':
            new_y += 1
        if dir == 'west':
            new_x -= 1
        if dir == 'east':
            new_x += 1
        self.room_ID2Coord[new_room_id] = (new_x, new_y)
        self.room_Coord2ID[(new_x, new_y)] = new_room_id
        # init 新房间的 exit_info
        self.room_exits_info[new_room_id] = {'north': None, 'south': None, 'west': None, 'east': None}
        # update 当前房间exit信息
        self.room_exits_info[self.current_room_id][dir] = ExitInfo(
            tiles=exit_infos[dir].tiles,
            exit_type=exit_infos[dir].exit_type,
            opened=exit_infos[dir].opened,
            dest=new_room_id,  # 通往的房间id
            start=self.current_room_id,
            is_reached=False,  # 是否已经到达过dest房间
            direction=dir,
        )

    def activate_switch(self, sym: SymbolicObs):
        """
        激活switch发生的房间转换逻辑
        每次switch:导致与当前房间相邻的所有exit通向新的房间
        """
        self.switched = True
        for dir, exitInfo in self.room_exits_info[self.current_room_id].items():
            if exitInfo is not None:
                # 该方向存在exit
                print('activate switch')
                print(dir)
                self.init_new_room(dir, sym)

    def activate_button(self):
        """激活button发生的逻辑"""
        # TODO
        pass

    def _bfs(self, start_room: int, dest_room: int) -> str | None:
        """找到start_room与dest_room之间的路径，返回路径下一步应该的方向"""
        if start_room == dest_room:
            print(f'start_room == dest_room {start_room}-{dest_room}')
            return None

        start = self.room_ID2Coord[start_room]

        dest = self.room_ID2Coord[dest_room]
        if dest_room == 2:
            print(f'start:{start} dest:{dest}')
        q = deque([start])
        parent: Dict[Pos, Tuple[Optional[Pos], Optional[str]]] = {
            start: (None, None)
        }

        while q:
            cur = q.popleft()
            curId = self.room_Coord2ID.get(cur, None)
            if curId is None:
                continue

            for nxt, dir in self.neighbors(cur):
                nid = self.room_Coord2ID.get(nxt, None)

                print(
                    f'dest_id : {dest_room} dest : {dest} nxt:{nxt} dir:{dir} nid:{nid} exit: {self.room_exits_info[curId]} ')
                if ((nid is None) or (self.room_exits_info[curId][dir] is None) or
                        ((self.room_exits_info[curId][dir].exit_type == 'locked_key') and (
                        not self.room_exits_info[curId][dir].opened) and (not self.has_key))):  # 如果隔壁房间不存在或无法通过
                    # 这里没有通往的房间 or 没有exit or 这个exit有锁且未打开且没有key
                    print('can not cross')
                    continue
                if nxt in parent:
                    print('parent node')
                    continue
                x, y = nxt
                print(nxt)
                parent[nxt] = (cur, dir)

                if nxt == dest:
                    # 回溯动作
                    dirs = []
                    p = nxt
                    while parent[p][0] is not None:
                        prev, dir = parent[p]
                        dirs.append(dir)
                        p = prev
                    dirs.reverse()
                    return dirs[0]

                q.append(nxt)

        return None

    def nearest(self, start: Pos, candidates: List[Pos]) -> Optional[Pos]:
        """从candidates中找到距离start最近的一个"""
        if not candidates:
            return None
        sx, sy = start
        return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

    def achieve_subgoal(self, subgoal: Subgoal | None, sym: SymbolicObs):
        """
        完成子任务后对planner的记忆进行更新
        内部会有二次检验是否完成子任务
        """
        print(f'cur_room:{self.current_room_id}')
        print(self.rooms)
        if subgoal is None:
            # 处理为空，也就是游戏刚刚开始
            if 0 in self.rooms['unexplored'] and self.current_room_id == 0:
                self.explore_room(sym)
                print(f'cur_room:{self.current_room_id}')
                print(self.rooms)
                return
        if subgoal.kind == 'go_exit':
            x, y = sym.player
            start = subgoal.start_room_id
            dest = subgoal.dest_room_id
            exit_dir = subgoal.exit_dir
            # 检查是否确实到达并通过exit
            flag = ((exit_dir == 'north' and y >= (ROOM_H - 1) // 2) or
                    (exit_dir == 'south' and y < (ROOM_H - 1) // 2) or
                    (exit_dir == 'west' and x >= (ROOM_W - 1) // 2) or
                    (exit_dir == 'east' and x < (ROOM_W - 1) // 2))
            if flag:
                self.room_exits_info[start][exit_dir].is_reached = True
                self.last_room_id = self.current_room_id
                self.current_room_id = dest
                # expored
                print(self.current_room_id, "achieved")
                # switch
                self.switched = False
                # 如果该房间还没探索过
                if dest in self.rooms['unexplored'] and self.current_room_id == dest:
                    self.explore_room(sym)

        if subgoal.kind == 'switch':
            self.activate_switch(sym)