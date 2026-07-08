import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
from nesylink.core.rendering import sprites as sp
from nesylink.core.rendering.sprites import CHEST_OPEN_INNER
# from agent import SymbolicObs, ExitInfo, ROOM_W, ROOM_H

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

#grid : [8 , 10]
ROOM_W = 10
ROOM_H = 8


Pos = Tuple[int, int]
pxPos = Tuple[int, int]  # 像素坐标 (0~159, 0~127)

def blank_tile():
    frame = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    sp.draw_floor(frame, 0, 0)
    return frame


def make_static_templates():
    templates = []

    # floor
    f = blank_tile()
    templates.append((EMPTY, "floor", f))

    # wall
    f = blank_tile()
    sp.draw_wall(f, 0, 0)
    templates.append((WALL, "wall", f))

    # gap
    f = blank_tile()
    sp.draw_gap(f, 0, 0)
    templates.append((GAP, "gap", f))

    # bridge
    f = blank_tile()
    sp.draw_bridge(f, 0, 0)
    templates.append((BRIDGE, "bridge", f))

    # chests: 三种 loot 外观不同，但都归为 CHEST
    for loot in ["key", "gold", "heal", "item", ""]:
        f = blank_tile()
        sp.draw_chest(f, 0, 0, opened=False, loot_kind=loot)
        templates.append((CHEST, f"chest_{loot}", f))

    # lcd : chests_opened: 打开了的chest相当于墙,归于WALL类
    for loot in ["key", "gold", "heal", "item", ""]:
        f = blank_tile()
        sp.draw_chest(f, 0, 0, opened=True, loot_kind=loot)
        templates.append((WALL, f"chest_opened_{loot}", f))


    # button
    for pressed in [False, True]:
        f = blank_tile()
        sp.draw_button(f, 0, 0, pressed=pressed)
        templates.append((BUTTON, f"button_{pressed}", f))

    # switch
    for activated in [False, True]:
        f = blank_tile()
        sp.draw_switch(f, 0, 0, activated=activated)
        templates.append((SWITCH, f"switch_{activated}", f))

    # trap
    f = blank_tile()
    sp.draw_trap(f, 0, 0)
    templates.append((TRAP, "trap", f))

    # abyss
    f = blank_tile()
    sp.draw_abyss(f, 0, 0)
    templates.append((TRAP, "abyss", f))

    # npc
    f = blank_tile()
    sp.draw_npc(f, 0, 0, sp.HIGHLIGHT)
    templates.append((NPC, "npc", f))

    return templates

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

#识别静态item
class StaticTileClassifier:
    def __init__(self):
        self.templates = make_static_templates()

    def classify_tile(self, patch):
        best_label = EMPTY
        best_name = "floor"
        best_score = 1e18

        for label, name, tmpl in self.templates:
            score = mse(patch, tmpl)
            if score < best_score:
                best_score = score
                best_label = label
                best_name = name

        return best_label, best_name, best_score
    

def sprite_to_template(sprite, palette):
    h = len(sprite)
    w = len(sprite[0])

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    for yy, row in enumerate(sprite):
        for xx, key in enumerate(row):
            color = palette.get(key)
            if color is not None:
                rgb[yy, xx] = color
                mask[yy, xx] = True

    return rgb, mask


def masked_mse(patch, rgb, mask):
    if mask.sum() == 0:
        return 1e18
    diff = patch.astype(np.float32)[mask] - rgb.astype(np.float32)[mask]
    return float(np.mean(diff * diff))

#识别player
class PlayerDetector:
    def __init__(self):
        self.templates = []

        for facing, sprite in sp.PLAYER_SPRITES.items():
            rgb, mask = sprite_to_template(sprite, sp.PLAYER_PALETTE)
            self.templates.append((facing, rgb, mask))

    def detect(self, frame):
        H, W = frame.shape[:2]

        best = {
            "score": 1e18,
            "xy": None,
            "facing": None,
        }

        for facing, rgb, mask in self.templates:
            h, w = rgb.shape[:2]

            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    patch = frame[y:y+h, x:x+w, :]
                    score = masked_mse(patch, rgb, mask)

                    if score < best["score"]:
                        best = {
                            "score": score,
                            "xy": (x, y),
                            "facing": facing,
                        }

        if best["score"] > 1000:
            return None

        x, y = best["xy"]

        # player 的 position_px 是 sprite 左上角。
        # 换算成 tile 时，用脚底中心更稳。
        cx = x + 8
        cy = y + 12

        tx = max(0, min(9, cx // 16))
        ty = max(0, min(7, cy // 16))

        return {
            "tile": (int(tx), int(ty)),
            "position_px": (x, y),
            "facing": best["facing"],
            "score": best["score"],
        }
    

from nesylink.core.rendering.renderer import MONSTER_COLORS

#识别monster
class MonsterDetector:
    def __init__(self):
        self.templates = []

        for monster_type, sprite in sp.MONSTER_SPRITES.items():
            color = MONSTER_COLORS[monster_type]
            palette = {
                "O": sp.OUTLINE,
                "M": color,
                "H": sp.MONSTER_DARK,
                "E": sp.MONSTER_EYE,
            }
            rgb, mask = sprite_to_template(sprite, palette)
            self.templates.append((monster_type, rgb, mask))

    def detect_all(self, frame):
        H, W = frame.shape[:2]
        detections = []

        for monster_type, rgb, mask in self.templates:
            h, w = rgb.shape[:2]

            for y in range(0, H - h + 1):
                for x in range(0, W - w + 1):
                    patch = frame[y:y+h, x:x+w, :]
                    score = masked_mse(patch, rgb, mask)

                    if score < 1000:
                        cx = x + 8
                        cy = y + 8
                        tx = max(0, min(9, cx // 16))
                        ty = max(0, min(7, cy // 16))
                        detections.append((score, monster_type, int(tx), int(ty), x, y))

        # 同一个 tile 去重
        best = {}
        for score, monster_type, tx, ty, x, y in detections:
            key = (tx, ty)
            if key not in best or score < best[key][0]:
                best[key] = (score, monster_type, x, y)

        return [
            {
                "tile": tile,
                "type": v[1],
                "position_px": (v[2], v[3]),
                "score": v[0],
            }
            for tile, v in best.items()
        ]



@dataclass
class SymbolicObs:
    grid: np.ndarray                    # shape: (8, 10)
    player: Optional[Pos] = None
    facing: str = "up"
    monsters: List[Pos] = field(default_factory=list)
    chests: List[Pos] = field(default_factory=list)
    exits: List[Pos] = field(default_factory=list)

    exit_infos: List['ExitInfo'] = field(default_factory=list)
    traps: List[Pos] = field(default_factory=list)
    buttons: List[Pos] = field(default_factory=list)
    switches: List[Pos] = field(default_factory=list)

    
# class PixelPerception:
#     def __init__(self):
#         self.static_clf = StaticTileClassifier()
#         self.player_detector = PlayerDetector()
#         self.monster_detector = MonsterDetector()

#     def __call__(self, obs):
#         frame = np.asarray(obs)

#         # 防止误传 render()，只取地图区域
#         frame = frame[:128, :160, :3]

#         grid = np.zeros((8, 10), dtype=np.int64)

#         # 1. 静态 tile 分类
#         for y in range(8):
#             for x in range(10):
#                 patch = frame[y*16:(y+1)*16, x*16:(x+1)*16, :]
#                 label, name, score = self.static_clf.classify_tile(patch)

#                 # 这里阈值可以设严格一点。
#                 # 如果 score 太大，默认 floor，避免误判。
#                 if score < 500:
#                     grid[y, x] = label
#                 else:
#                     grid[y, x] = EMPTY

#         # 2. 玩家覆盖修正
#         player_info = self.player_detector.detect(frame)
#         player = None
#         facing = "down"

#         if player_info is not None:
#             player = player_info["tile"]
#             facing = player_info["facing"]
#             px, py = player
#             grid[py, px] = PLAYER

#         # 3. 怪物覆盖修正
#         monsters = []
#         for m in self.monster_detector.detect_all(frame):
#             tx, ty = m["tile"]
#             monsters.append((tx, ty))
#             grid[ty, tx] = MONSTER

#         return self.grid_to_symbolic(grid, player, facing, monsters)

#     def grid_to_symbolic(self, grid, player, facing, monsters):
#         chests = []
#         traps = []
#         buttons = []
#         switches = []
#         npcs = []
#         gaps = []
#         bridges = []
#         exits = []

#         for y in range(8):
#             for x in range(10):
#                 v = int(grid[y, x])

#                 if v == CHEST:
#                     chests.append((x, y))
#                 elif v == TRAP:
#                     traps.append((x, y))
#                 elif v == BUTTON:
#                     buttons.append((x, y))
#                 elif v == SWITCH:
#                     switches.append((x, y))
#                 elif v == NPC:
#                     npcs.append((x, y))
#                 elif v == GAP:
#                     gaps.append((x, y))
#                 elif v == BRIDGE:
#                     bridges.append((x, y))
#                 elif v == EXIT:
#                     exits.append((x, y))

#         return SymbolicObs(
#             grid=grid,
#             player=player,
#             facing=facing,
#             monsters=monsters,
#             chests=chests,
#             exits=exits,
#             traps=traps,
#             buttons=buttons,
#             switches=switches,
#         )
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

        # exit_infos = self.exit_detector.detect(frame)

        # exits = []
        # for e in exit_infos:
        #     x, y = e["tile"]
        #     type_ = e.get("exit_type", "unknown")
        #     opened = e.get("opened", False)
        #     exits.append((x, y, type_, opened))
        #     grid[y, x] = EXIT

        exit_infos_raw = self.exit_detector.detect(frame)

        exits: List[Pos] = []
        exit_infos: List[ExitInfo] = []

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
                    exits.append((x, y))
                    grid[y, x] = EXIT

        # 2. 玩家覆盖修正
        player_info = self.player_detector.detect(frame)
        player = None
        facing = "down"

        if player_info is not None:
            player = player_info["tile"]
            facing = player_info["facing"]
            px, py = player
            grid[py, px] = PLAYER

        # 3. 怪物覆盖修正
        monsters = []
        for m in self.monster_detector.detect_all(frame):
            tx, ty = m["tile"]
            monsters.append((tx, ty))
            grid[ty, tx] = MONSTER

        return self.grid_to_symbolic(grid, player, facing, monsters, exits, exit_infos)

    def grid_to_symbolic(self, grid, player, facing, monsters, exits_hint=None, exit_infos_hint=None):
        chests = []
        traps = []
        buttons = []
        switches = []
        npcs = []
        gaps = []
        bridges = []
        exits = list(exits_hint) if exits_hint is not None else []
        exit_infos = list(exit_infos_hint) if exit_infos_hint is not None else []

        for y in range(8):
            for x in range(10):
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

        return SymbolicObs(
            grid=grid,
            player=player,
            facing=facing,
            monsters=monsters,
            chests=chests,
            exits=exits,
            exit_infos=exit_infos,
            traps=traps,
            buttons=buttons,
            switches=switches,
        )

# from nesylink.core.rendering import sprites as sp
from nesylink.core.constants import (
    COLOR_EXIT_NORMAL,
    COLOR_EXIT_LOCKED,
    COLOR_EXIT_CONDITIONAL,
)

def make_exit_patch(direction: str, exit_type: str, opened: bool = False):
    """
    生成和真实渲染一致的出口模板。
    direction: north/south/west/east
    exit_type: normal/locked_key/conditional
    """

    if direction in {"west", "east"}:
        # 竖向门：1列×2行，大小 16×32
        patch = np.zeros((32, 16, 3), dtype=np.uint8)
        sp.draw_floor(patch, 0, 0)
        sp.draw_floor(patch, 0, 1)
        tiles = ((0, 0), (0, 1))
    else:
        # 横向门：2列×1行，大小 32×16
        patch = np.zeros((16, 32, 3), dtype=np.uint8)
        sp.draw_floor(patch, 0, 0)
        sp.draw_floor(patch, 1, 0)
        tiles = ((0, 0), (1, 0))

    if exit_type == "locked_key":
        color = COLOR_EXIT_LOCKED
    elif exit_type == "conditional":
        color = COLOR_EXIT_CONDITIONAL
    else:
        color = COLOR_EXIT_NORMAL

    sp.draw_exit(patch, tiles, exit_type, color, opened=opened)
    return patch

@dataclass
class ExitInfo:
    tiles: List[Pos]              # 这个出口占据的两个 tile
    direction: str                # north/south/west/east
    exit_type: str = "unknown"    # normal/locked_key/conditional/unknown
    opened: bool = False
    score: float = 0.0

    @property
    def representative(self) -> Pos:
        return self.tiles[0]


class ExitDetector:
    def __init__(self):
        self.templates = []

        for direction in ["north", "south", "west", "east"]:
            for exit_type in ["normal", "locked_key", "conditional"]:
                for opened in [False, True]:
                    patch = make_exit_patch(direction, exit_type, opened)
                    self.templates.append({
                        "direction": direction,
                        "exit_type": exit_type,
                        "opened": opened,
                        "patch": patch,
                    })

    def candidate_regions(self, frame):
        """
        只在固定边缘位置找出口。
        返回: direction, representative_tile, image_patch
        """

        cands = []

        # north/south 门通常在中间两格附近，但为了泛化，沿边缘滑动 2-tile 窗口
        for x in range(0, ROOM_W - 1):
            # north: y=0, 2列×1行
            patch = frame[0:16, x*16:(x+2)*16, :]
            cands.append(("north", (x, 0), patch))

            # south: y=7
            patch = frame[7*16:8*16, x*16:(x+2)*16, :]
            cands.append(("south", (x, 7), patch))

        for y in range(0, ROOM_H - 1):
            # west: x=0, 1列×2行
            patch = frame[y*16:(y+2)*16, 0:16, :]
            cands.append(("west", (0, y), patch))

            # east: x=9
            patch = frame[y*16:(y+2)*16, 9*16:10*16, :]
            cands.append(("east", (9, y), patch))

        return cands

    def detect(self, frame, threshold=500.0):
        exits = []
        best_by_region = {}

        for direction, tile, patch in self.candidate_regions(frame):
            best = None

            for tmpl in self.templates:
                if tmpl["direction"] != direction:
                    continue
                if tmpl["patch"].shape != patch.shape:
                    continue

                score = mse(patch, tmpl["patch"])
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "direction": direction,
                        "tile": tile,
                        "exit_type": tmpl["exit_type"],
                        "opened": tmpl["opened"],
                    }

            if best is not None and best["score"] < threshold:
                key = (direction, tile)
                best_by_region[key] = best

        # 合并：同一方向可能滑动出多个相邻候选，只保留最低分
        best_by_direction = {}
        for item in best_by_region.values():
            d = item["direction"]
            if d not in best_by_direction or item["score"] < best_by_direction[d]["score"]:
                best_by_direction[d] = item

        for item in best_by_direction.values():
            item = dict(item)  # copy
            item["tiles"] = self.exit_tiles_from_rep(item["direction"], item["tile"])
            exits.append(item)

        return exits
    
    def exit_tiles_from_rep(self, direction: str, tile: Pos) -> List[Pos]:
        x, y = tile

        if direction in {"north", "south"}:
            return [(x, y), (x + 1, y)]

        if direction in {"west", "east"}:
            return [(x, y), (x, y + 1)]

        return [tile]
