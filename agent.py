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

# lcd : 任务中可能发生的事件，复制到这里方便写代码
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

# lcd
def nearest_px(start: pxPos, candidates: List[pxPos]) -> Optional[pxPos]:
    """nearest函数的像素级别版本"""
    if not candidates:
        return None
    sx, sy = start
    return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

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


from Dataclass import Candidate
from symbolicPlanner import SymbolicPlanner

from optionController import OptionController

from safetyShield import SafetyShield

class Policy:
    def __init__(self) -> None:
        self.perception = PixelPerception()
        self.belief = BeliefState()
        self.planner = SymbolicPlanner()
        self.controller = OptionController()
        self.shield = SafetyShield()

        self.action_queue: Deque[int] = deque()
        self.current_subgoal: Optional[Subgoal] = None

        #维护sym
        self.last_action = ACTION_NOOP
        self.sym: Optional[SymbolicObs] = None
        self.perception_interval = 100  # 先用 4，稳定后可以改成 8

        self.force_exit_action: Optional[int] = None
        self.force_exit_steps: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.belief.reset(task_id=task_id)
        self.action_queue.clear()
        self.current_subgoal = None

        self.sym = None
        self.force_exit_action = None
        self.force_exit_steps = 0

    def act(self, obs, info=None) -> int:
        #每次act之前维护好sym
        if self.sym is not None:
            self.update_sym(self.last_action)
        # 是否需要再次识别图片
        need_vision = (
                self.sym is None
                or not self.action_queue
                or self.belief.step % self.perception_interval == 0
                or self.sym.monsters  # 如果有monster要继续vision
        )

        if need_vision:
            sym = self.perception(obs)
            self.sym = sym
            self.belief.update(sym, info)
        else:
            # 如果不识别图片，要推断变化值：facing,player_px,player
            sym = self.sym
            self.belief.step += 1

        replanned = False
        # 附近有monster
        encounter_monster = is_encounter_monster(sym, )
        if encounter_monster:
            replanned = True

        # 3. 判断是否需要重新规划
        if self.need_replan(sym, obs, info, replanned):
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
                "player_px=", sym.player_px,
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

        # 5. 安全过滤
        action = self.shield.filter(raw_action, sym, self.belief)

        if self.belief.step % 1 == 0 or replanned:
            # lcd : 增加调试信息
            print(f"<<<<<<<<<<<< step: {self.belief.step} >>>>>>>>>>>>")
            print(
                "[INFO]",
                "player=", info['agent'],
                "monster=", info["entities"]["monsters_remaining"],
                "event=", info["events"]["records"]
            )
            print(
                "[ACT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "player_px=", sym.player_px,
                "monster=", sym.monsters,
                "monster_px=", sym.monsters_px,
                "chests=",sym.chests,
                "raw=", raw_action,
                "safe=", action,
                "queue_left=", len(self.action_queue),
                "stuck=", self.belief.stuck_count,
                "subgoal=", self.current_subgoal,
            )

        self.belief.last_action = action
        # 如果不识别图片，要推断变化值：facing,player_px,player
        self.last_action = action
        return int(action)

    def update_sym(self, action: int):
        """
        如果不识别图片，要推断变化值：facing,player_px,player
        有时候player会对图片进行遮挡，导致exit无法识别，可以使用planner的记忆进行处理
        一般action取self.last_action
        一般在self.act开头调用，保障sym无误
        """
        dx, dy = 0, 0
        if action == ACTION_LEFT:
            self.sym.facing = 'left'
            dx -= 1
        if action == ACTION_RIGHT:
            self.sym.facing = 'right'
            dx += 1
        if action == ACTION_UP:
            self.sym.facing = 'up'
            dy -= 1
        if action == ACTION_DOWN:
            self.sym.facing = 'down'
            dy += 1
        x, y = self.sym.player_px
        # 如果能保障action合法，则坐标必然在有 0<=x<=144 and 0<=y<=112 遇到exit应该取模
        self.sym.player_px = (((x + dx) % 145), ((y + dy) % 113))
        cx = (x + dx) % 145 + 8
        cy = (y + dy) % 113 + 12

        tx = max(0, min(9, cx // 16))
        ty = max(0, min(7, cy // 16))

        self.sym.player = (tx, ty)

        # 测试发现agent有概率遮住exit,需要利用房间记忆对sym的exits进行维护
        cur_room = self.planner.current_room_id
        print(cur_room, 'cur_room')
        self.sym.exits = []
        for dir, exitInfo in self.planner.room_exits_info[cur_room].items():
            if (exitInfo is None) or (exitInfo.tiles is None):
                continue
            tiles = exitInfo.tiles
            self.sym.exit_infos[dir] = exitInfo  # 维护exit_infos
            for t in tiles:
                if t not in self.sym.exits:
                    self.sym.exits.append(t)  # 维护exits
        print(self.sym.exits)

    def need_replan(self, sym: SymbolicObs, obs, info=None, force_replan=False) -> bool:
        """判断是否需要重新规划"""
        # lcd : 强制replan,发生特定事情需要，比如有monster靠近
        if force_replan:
            return True

        # 没有动作了，必须重新规划,并对planner进行更新
        if not self.action_queue:
            self.planner.achieve_subgoal(self.current_subgoal, sym)
            return True

        # 识别不到玩家，先不继续盲走
        if sym.player is None:
            self.action_queue.clear()
            return True

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