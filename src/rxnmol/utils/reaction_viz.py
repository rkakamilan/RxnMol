"""Reaction route tree visualization.

Single source of truth for tree-like synthesis route rendering.
Used by both the notebook (_reaction_viz.ipynb) and the CLI script
(scripts/visualize_routes.py).
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _get_fonts(size_normal=13, size_small=11, size_bold=14, size_title=18, size_arrow=24):
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size_normal)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size_small)
        font_bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size_bold)
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size_title)
        font_arrow = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size_arrow)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
        font_bold = font
        font_title = font
        font_arrow = font
    return font, font_small, font_bold, font_title, font_arrow


# ---------------------------------------------------------------------------
# SMILES → PIL Image
# ---------------------------------------------------------------------------

def smi_to_img(smi, size=(300, 300)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        img = Image.new('RGB', size, 'white')
        d = ImageDraw.Draw(img)
        d.text((10, size[1] // 2 - 10), 'Invalid SMILES', fill='red')
        return img
    return Draw.MolToImage(mol, size=size)


# ---------------------------------------------------------------------------
# Route parsing
# ---------------------------------------------------------------------------

def parse_rxnmol_route(genotype_str, intermediates_str):
    try:
        frags = ast.literal_eval(genotype_str) if isinstance(genotype_str, str) else list(genotype_str)
    except Exception:
        frags = []
    try:
        inters = ast.literal_eval(intermediates_str) if isinstance(intermediates_str, str) else list(intermediates_str)
    except Exception:
        inters = []
    return frags, inters


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class ReactionTreeNode:
    """A node in the synthesis route tree."""
    smiles: str
    label: str
    node_type: str          # 'scaffold', 'fragment', 'intermediate', 'final'
    left: Optional['ReactionTreeNode'] = None
    right: Optional['ReactionTreeNode'] = None


def build_reaction_tree(fragments: List[str], intermediates: List[str]) -> ReactionTreeNode:
    """
    Build a left-heavy binary tree from fragments and intermediates.

    Tree shape for [F1, F2, F3, F4]:
              Final
             /     \\
           I2       F4
          /   \\
        I1     F3
       /  \\
      F1   F2
    """
    n = len(fragments)
    if n == 0:
        return ReactionTreeNode(smiles='', label='Empty', node_type='scaffold')
    if n == 1:
        return ReactionTreeNode(smiles=fragments[0], label='Fragment 1', node_type='scaffold')

    current = ReactionTreeNode(smiles=fragments[0], label='Fragment 1', node_type='scaffold')

    for i in range(1, n):
        frag_node = ReactionTreeNode(
            smiles=fragments[i],
            label=f'Fragment {i + 1}',
            node_type='fragment',
        )
        is_final = (i == n - 1)
        inter_smi = intermediates[i] if i < len(intermediates) else ''
        current = ReactionTreeNode(
            smiles=inter_smi,
            label='Final Product' if is_final else f'Intermediate {i}',
            node_type='final' if is_final else 'intermediate',
            left=current,
            right=frag_node,
        )
    return current


# ---------------------------------------------------------------------------
# Layout engine (proportional allocation)
# ---------------------------------------------------------------------------

def _subtree_spread(node, unit_size, gap):
    """Compute spread size (pixels) for the subtree rooted at node."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return unit_size
    lw = _subtree_spread(node.left, unit_size, gap)
    rw = _subtree_spread(node.right, unit_size, gap)
    return lw + gap + rw


def _tree_depth(node):
    if node is None:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def compute_tree_layout(
    root: ReactionTreeNode,
    mol_size: Tuple[int, int] = (220, 180),
    h_gap: int = 15,
    v_gap: int = 35,
    title_h: int = 40,
    label_h: int = 20,
    direction: str = 'bottom-up',
) -> Tuple[Dict[int, Tuple[int, int]], int, int]:
    """
    Compute pixel positions for all nodes.

    direction:
        'bottom-up'  : fragments at bottom, product at top (default)
        'top-down'   : fragments at top, product at bottom
        'left-right' : fragments at left, product at right

    Returns: (positions, total_width, total_height)
    """
    mol_w, mol_h = mol_size
    max_depth = _tree_depth(root)
    padding = 20

    if direction in ('bottom-up', 'top-down'):
        tree_spread = _subtree_spread(root, mol_w, h_gap)
        level_stride = mol_h + label_h + v_gap

        total_w = tree_spread + 2 * padding
        total_h = title_h + max_depth * level_stride - v_gap + padding

        positions = {}

        def assign(node, s_left, s_right, depth):
            if node is None:
                return
            if direction == 'bottom-up':
                y = title_h + depth * level_stride
            else:  # top-down
                y = title_h + (max_depth - 1 - depth) * level_stride

            if node.left is None and node.right is None:
                cx = (s_left + s_right) // 2
                positions[id(node)] = (cx - mol_w // 2, y)
                return

            lw = _subtree_spread(node.left, mol_w, h_gap) if node.left else 0
            rw = _subtree_spread(node.right, mol_w, h_gap) if node.right else 0
            total = lw + h_gap + rw
            avail = s_right - s_left
            l_end = s_left + int(avail * lw / total)
            r_start = s_left + int(avail * (lw + h_gap) / total)
            assign(node.left, s_left, l_end, depth + 1)
            assign(node.right, r_start, s_right, depth + 1)
            cx = (s_left + s_right) // 2
            positions[id(node)] = (cx - mol_w // 2, y)

        assign(root, padding, padding + tree_spread, 0)

    else:  # left-right: spread = vertical, level = horizontal
        cell_h = mol_h + label_h
        tree_spread = _subtree_spread(root, cell_h, v_gap)
        level_stride = mol_w + h_gap

        total_w = padding + max_depth * level_stride - h_gap + padding
        total_h = title_h + tree_spread + padding

        positions = {}

        def assign(node, s_top, s_bottom, depth):
            if node is None:
                return
            x = padding + (max_depth - 1 - depth) * level_stride  # root at right

            if node.left is None and node.right is None:
                sc = (s_top + s_bottom) // 2
                positions[id(node)] = (x, title_h + sc - cell_h // 2)
                return

            lsw = _subtree_spread(node.left, cell_h, v_gap) if node.left else 0
            rsw = _subtree_spread(node.right, cell_h, v_gap) if node.right else 0
            total = lsw + v_gap + rsw
            avail = s_bottom - s_top
            l_end = s_top + int(avail * lsw / total)
            r_start = s_top + int(avail * (lsw + v_gap) / total)
            assign(node.left, s_top, l_end, depth + 1)
            assign(node.right, r_start, s_bottom, depth + 1)
            sc = (s_top + s_bottom) // 2
            positions[id(node)] = (x, title_h + sc - cell_h // 2)

        assign(root, 0, tree_spread, 0)

    return positions, total_w, total_h


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLORS = {
    'scaffold':     '#1a7a1a',   # green
    'fragment':     '#cc6600',   # orange
    'intermediate': '#0055aa',   # blue
    'final':        '#cc0000',   # red
    'connector':    '#555555',
}

LABEL_H = 20  # approx height of label text


def _draw_node(canvas, draw, node, x, y, mol_size, fonts):
    """Draw molecule image + label for one node."""
    font, font_small, font_bold, font_title, font_arrow = fonts
    mol_w, mol_h = mol_size
    color = COLORS.get(node.node_type, 'black')

    img = smi_to_img(node.smiles, size=mol_size)
    canvas.paste(img, (x, y))

    if node.node_type == 'final':
        draw.rectangle([x - 2, y - 2, x + mol_w + 2, y + mol_h + 2],
                       outline='red', width=3)

    label_x = x + mol_w // 2
    label_y = y + mol_h + 2
    label_font = font_bold if node.node_type == 'final' else font
    draw.text((label_x, label_y), node.label, fill=color, font=label_font, anchor='mt')


def _draw_lines(draw, node, positions, mol_size, fonts, direction='bottom-up'):
    """Draw connector lines (pass 1 — behind molecules)."""
    if node.left is None and node.right is None:
        return
    font, font_small, font_bold, font_title, font_arrow = fonts
    mol_w, mol_h = mol_size
    lc = COLORS['connector']
    lw = 2

    for child in (node.left, node.right):
        if child is not None:
            _draw_lines(draw, child, positions, mol_size, fonts, direction)

    px, py = positions[id(node)]
    cpos = [positions[id(c)] for c in (node.left, node.right) if c is not None]
    if not cpos or len(cpos) < 2:
        return

    pcx = px + mol_w // 2

    if direction == 'bottom-up':
        parent_edge = py + mol_h + LABEL_H
        cpts = [(cx + mol_w // 2, cy) for cx, cy in cpos]
        jy = parent_edge + (cpts[0][1] - parent_edge) // 2
        c1x, _ = cpts[0]; c2x, _ = cpts[1]
        draw.line([(c1x, cpts[0][1]), (c1x, jy)], fill=lc, width=lw)
        draw.line([(c2x, cpts[1][1]), (c2x, jy)], fill=lc, width=lw)
        draw.line([(c1x, jy), (c2x, jy)], fill=lc, width=lw)

    elif direction == 'top-down':
        parent_edge = py
        cpts = [(cx + mol_w // 2, cy + mol_h + LABEL_H) for cx, cy in cpos]
        jy = cpts[0][1] + (parent_edge - cpts[0][1]) // 2
        c1x, c1y = cpts[0]; c2x, c2y = cpts[1]
        draw.line([(c1x, c1y), (c1x, jy)], fill=lc, width=lw)
        draw.line([(c2x, c2y), (c2x, jy)], fill=lc, width=lw)
        draw.line([(c1x, jy), (c2x, jy)], fill=lc, width=lw)

    elif direction == 'left-right':
        parent_edge = px
        pcy = py + mol_h // 2
        cpts = [(cx + mol_w, cy + mol_h // 2) for cx, cy in cpos]
        jx = cpts[0][0] + (parent_edge - cpts[0][0]) // 2
        c1x, c1y = cpts[0]; c2x, c2y = cpts[1]
        draw.line([(c1x, c1y), (jx, c1y)], fill=lc, width=lw)
        draw.line([(c2x, c2y), (jx, c2y)], fill=lc, width=lw)
        draw.line([(jx, c1y), (jx, c2y)], fill=lc, width=lw)


def _draw_arrows(draw, node, positions, mol_size, fonts, direction='bottom-up'):
    """Draw arrow glyphs on stems (pass 2 — on top of molecules)."""
    if node.left is None and node.right is None:
        return
    font, font_small, font_bold, font_title, font_arrow = fonts
    mol_w, mol_h = mol_size
    lc = COLORS['connector']

    for child in (node.left, node.right):
        if child is not None:
            _draw_arrows(draw, child, positions, mol_size, fonts, direction)

    px, py = positions[id(node)]
    cpos = [positions[id(c)] for c in (node.left, node.right) if c is not None]
    if not cpos or len(cpos) < 2:
        return

    pcx = px + mol_w // 2

    if direction == 'bottom-up':
        parent_edge = py + mol_h + LABEL_H
        cpts = [(cx + mol_w // 2, cy) for cx, cy in cpos]
        jy = parent_edge + (cpts[0][1] - parent_edge) // 2
        mid_y = (jy + parent_edge) // 2
        draw.text((pcx, mid_y), '\u2191', fill=lc, font=font_arrow, anchor='mm')

    elif direction == 'top-down':
        parent_edge = py
        cpts = [(cx + mol_w // 2, cy + mol_h + LABEL_H) for cx, cy in cpos]
        jy = cpts[0][1] + (parent_edge - cpts[0][1]) // 2
        mid_y = (jy + parent_edge) // 2
        draw.text((pcx, mid_y), '\u2193', fill=lc, font=font_arrow, anchor='mm')

    elif direction == 'left-right':
        parent_edge = px
        pcy = py + mol_h // 2
        cpts = [(cx + mol_w, cy + mol_h // 2) for cx, cy in cpos]
        jx = cpts[0][0] + (parent_edge - cpts[0][0]) // 2
        mid_x = (jx + parent_edge) // 2 + 5
        draw.text((mid_x, pcy), '\u2192', fill=lc, font=font_arrow, anchor='mm')


def draw_reaction_tree(
    genotype_str,
    intermediates_str,
    final_smi,
    objective=None,
    task_name=None,
    sa_score=None,
    qed_score=None,
    mol_size=(220, 180),
    direction='left-right',
    save_path=None,
    dpi=150,
):
    """
    Draw a tree-like synthesis route visualization.

    direction:
        'bottom-up'  : fragments at bottom, product at top (default)
        'top-down'   : fragments at top, product at bottom
        'left-right' : fragments at left, product at right
    """
    frags, inters = parse_rxnmol_route(genotype_str, intermediates_str)
    if not frags:
        return smi_to_img(final_smi, size=mol_size)

    root = build_reaction_tree(frags, inters)
    positions, total_w, total_h = compute_tree_layout(
        root, mol_size=mol_size, direction=direction)

    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    # Scale fonts proportionally to molecule size
    scale = mol_size[0] / 220
    fonts = _get_fonts(
        size_normal=max(13, int(13 * scale)),
        size_small=max(11, int(11 * scale)),
        size_bold=max(14, int(14 * scale)),
        size_title=max(18, int(18 * scale)),
        size_arrow=max(24, int(24 * scale)),
    )

    # Title
    parts = []
    if task_name:
        parts.append(str(task_name))
    if objective is not None:
        parts.append(f'Obj: {abs(objective):.3f}')
    if sa_score is not None:
        parts.append(f'SA: {sa_score:.1f}')
    if qed_score is not None:
        parts.append(f'QED: {qed_score:.2f}')
    title = '  |  '.join(parts)
    draw.text((total_w // 2, 12), title, fill='black', font=fonts[3], anchor='mt')

    # Pass 1: lines (behind molecules)
    _draw_lines(draw, root, positions, mol_size, fonts, direction)

    # Pass 2: molecule images + labels
    def draw_all_nodes(node):
        if node is None:
            return
        x, y = positions[id(node)]
        _draw_node(canvas, draw, node, x, y, mol_size, fonts)
        draw_all_nodes(node.left)
        draw_all_nodes(node.right)
    draw_all_nodes(root)

    # Pass 3: arrows (on top of everything)
    _draw_arrows(draw, root, positions, mol_size, fonts, direction)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(save_path), dpi=(dpi, dpi))
        print(f'Saved: {save_path}')

    return canvas
