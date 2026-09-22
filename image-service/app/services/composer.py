"""Deterministic text/asset composition; diffusion supplies only the background."""
import hashlib
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from app.config.settings import settings
from app.schemas.composition import CompositionResult
from app.services.atomic import write_bytes


def normalized_zone_to_pixels(zone: dict) -> tuple[int, int, int, int]:
    x, y, w, h = (float(zone[key]) for key in ('x', 'y', 'width', 'height'))
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.000001 or y+h > 1.000001:
        raise ValueError('Invalid layout zone: outside page')
    result = (round(x*2480), round(y*3508), round((x+w)*2480), round((y+h)*3508))
    if result[0] < 119 or result[1] < 119 or result[2] > 2480-119 or result[3] > 3508-119:
        raise ValueError('Layout zone violates the 10 mm safe margin; correct the creative direction')
    return result


def load_font(size: int, bold: bool = False):
    suffix = '-Bold' if bold else ''
    for base in ['/usr/share/fonts/truetype/dejavu/DejaVuSans', '/usr/share/fonts/truetype/liberation2/LiberationSans']:
        path = Path(base + suffix + '.ttf')
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError('Required scalable fonts missing; install fonts-dejavu-core')


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split('\n'):
        current = ''
        for word in paragraph.split(' '):
            candidate = current + (' ' if current else '') + word
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
                if draw.textlength(word, font=font) > max_width:
                    raise ValueError('CONTENT_TOO_LONG: a word does not fit its reserved zone')
        lines.append(current)
    return lines


def draw_panel(image: Image.Image, texts: list[tuple[str, bool]], zone: dict,
               max_size: int, min_size: int = 42) -> None:
    left, top, right, bottom = normalized_zone_to_pixels(zone)
    padding = 24
    draw = ImageDraw.Draw(image)
    available_width = right-left-2*padding
    for size in range(max_size, min_size-1, -2):
        runs = []
        try:
            for text, bold in texts:
                font = load_font(size, bold)
                height = sum(font.getmetrics()) + 8
                for line in wrap_text(draw, text, font, available_width):
                    runs.append((line, font, height))
            if sum(run[2] for run in runs) <= bottom-top-2*padding:
                break
        except ValueError:
            pass
    else:
        raise ValueError('CONTENT_TOO_LONG: full text cannot fit at the 10 pt font floor')
    content_height = sum(run[2] for run in runs)
    panel_bottom = min(bottom, top + content_height + 2 * padding)
    # The panel protects contrast without turning the entire reserved zone
    # into a blank document.  It grows only as far as the composed content.
    draw.rounded_rectangle(
        (left, top, right, panel_bottom),
        radius=18,
        fill=(255, 255, 255, 238),
        outline=(212, 175, 55, 255),
        width=4,
    )
    y = top+padding
    for line, font, height in runs:
        draw.text((left+padding, y), line, font=font, fill='black', anchor='lt')
        y += height


def overlay_asset(canvas: Image.Image, asset_path: str | None, zone: dict | None) -> None:
    if not asset_path:
        return
    if not zone:
        raise ValueError('Supplied asset has no reserved layout zone')
    left, top, right, bottom = normalized_zone_to_pixels(zone)
    with Image.open(asset_path) as source:
        if source.format not in {'JPEG', 'PNG'}:
            raise ValueError('Supplied asset must be JPEG or PNG')
        image = source.convert('RGBA')
    image.thumbnail((right-left, bottom-top), Image.Resampling.LANCZOS)
    canvas.alpha_composite(image, dest=(left+(right-left-image.width)//2, top+(bottom-top-image.height)//2))


def compose_programme(reference_number: str, engine_id: str, direction_id: str,
                      background_path: str, brief: dict, layout_guidance: dict) -> CompositionResult:
    boxes = [normalized_zone_to_pixels(zone) for zone in layout_guidance.values() if isinstance(zone, dict) and 'x' in zone]
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                raise ValueError('Reserved layout zones overlap')
    with Image.open(background_path) as source:
        source.load()
        canvas = source.convert('RGBA').resize((2480, 3508), Image.Resampling.LANCZOS)
    title = [(brief['title'], True)]
    details = [str(brief[key]) for key in ['event_date', 'start_time', 'timezone', 'venue'] if brief.get(key)]
    if details:
        title.append((' | '.join(details), False))
    draw_panel(canvas, title, layout_guidance['title_zone'], max_size=80)
    rows = []
    for item in brief['programme']:
        time = item.get('time', item.get('start_time', ''))
        label = item.get('title', item.get('item', ''))
        rows.append((f'{time}  {label}', True))
        for key in ['speaker', 'description']:
            if item.get(key):
                rows.append((str(item[key]), False))
        if item.get('duration_minutes') is not None:
            rows.append((f"{item['duration_minutes']} min", False))
    draw_panel(canvas, rows, layout_guidance['programme_zone'], max_size=52)
    for name in ['headshot', 'logo']:
        overlay_asset(canvas, brief.get(name+'_path'), layout_guidance.get(name+'_zone'))
    path = settings.programme_data_path / reference_number / 'final' / f'{engine_id}-{direction_id.lower()}.png'
    buffer = io.BytesIO()
    canvas.convert('RGB').save(buffer, format='PNG', dpi=(300, 300))
    data = buffer.getvalue()
    write_bytes(path, data)
    return CompositionResult(reference_number=reference_number, engine_id=engine_id,
        direction_id=direction_id, background_path=background_path, output_path=str(path),
        sha256=hashlib.sha256(data).hexdigest(), width=2480, height=3508, dpi=300)
