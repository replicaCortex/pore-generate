"""Модуль для процедурной генерации изображений с порами."""

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_WHITE_COLOR = 255
_BLACK_COLOR = 0
_BINARY_THRESHOLD = 127

_IMAGE_MARGIN_FACTOR = 0.1

_PORE_PLACEMENT_ATTEMPTS = 200
_MAX_TOTAL_ATTEMPTS_MULTIPLIER = 1

_MIN_DISTANCE_LARGE = 8
_MIN_DISTANCE_MEDIUM = 5
_MIN_DISTANCE_SMALL = 3

_MORPH_KERNEL_ELLIPSE_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_MORPH_KERNEL_ELLIPSE_5X5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class PoreGenerator:
    """Генерирует бинарные изображения с реалистичными порами.

    Класс управляет всем процессом создания одного изображения: от генерации
    отдельных пор разного размера и формы до их размещения на холсте
    с учетом коллизий.

    Attributes:
        _width: Ширина генерируемых изображений.
        _height: Высота генерируемых изображений.
        _config: Словарь с полной конфигурацией.
    """

    def __init__(self, config: dict[str, Any]):
        """Инициализирует генератор пор.

        Args:
            config: Словарь конфигурации, содержащий `image_settings` и
                `pore_settings`.
        """
        image_settings = config.get("image_settings", {})
        self._width = image_settings.get("width", 512)
        self._height = image_settings.get("height", 512)
        self._config = config

        self._pore_data = []

    def generate_image(self) -> np.ndarray:
        """Создает одно изображение с порами всех сконфигурированных типов.

        Returns:
            Сгенерированное изображение в виде NumPy массива.
        """
        image = np.full((self._height, self._width), _WHITE_COLOR, dtype=np.uint8)
        occupied_mask = np.zeros((self._height, self._width), dtype=bool)

        self._pore_data = []

        pore_size_categories = ["large_pores", "medium_pores", "small_pores"]
        pore_type_categories = [
            "single",
            "weakly_overlapping",
            "strongly_overlapping",
            "defective",
        ]

        for size_cat in pore_size_categories:
            for type_cat in pore_type_categories:
                self._add_pores(image, occupied_mask, size_cat, type_cat)

        return image

    def _add_pores(
        self,
        image: np.ndarray,
        occupied_mask: np.ndarray,
        size_category: str,
        type_category: str,
    ) -> None:
        """Добавляет на изображение поры указанного размера и типа."""
        settings = (
            self._config.get("pore_settings", {})
            .get(size_category, {})
            .get(type_category)
        )

        if not settings:
            return

        count = random.randint(*settings.get("count_range", (0, 0)))
        current_radius_mean = settings.get("radius_mean", 10.0)

        placed_count = 0
        max_total_attempts = count * _MAX_TOTAL_ATTEMPTS_MULTIPLIER
        total_attempts = 0

        while placed_count < count and total_attempts < max_total_attempts:
            total_attempts += 1

            # Генерируем радиус для исходного круга
            std_dev = np.sqrt(current_radius_mean)
            min_r = max(3, int(current_radius_mean - 3 * std_dev))
            max_r = int(current_radius_mean + 3 * std_dev)
            original_radius = float(
                np.clip(np.random.poisson(lam=current_radius_mean), min_r, max_r)
            )

            # Генерируем деформированную пору
            pore_canvas = self._generate_single_pore_canvas_with_radius(
                settings, original_radius
            )

            position = self._find_valid_placement(
                pore_canvas, occupied_mask, size_category
            )

            if position:
                x, y = position
                self._place_on_image(
                    image,
                    occupied_mask,
                    pore_canvas,
                    x,
                    y,
                    size_category,
                    type_category,
                    original_radius,  # Передаем исходный радиус
                )
                placed_count += 1
            elif placed_count < count * 0.5:
                current_radius_mean = max(3.0, current_radius_mean * 0.95)

    def _generate_single_pore_canvas_with_radius(
        self, settings: dict[str, Any], radius: float
    ) -> np.ndarray:
        """Создает холст с одной порой, применяя все трансформации."""
        pore_type = random.choice(["smooth", "normal", "rough"])
        pore_canvas = self._create_realistic_pore(int(radius), pore_type)

        if settings.get("stretch_enabled", False):
            stretch_range = settings.get("stretch_factor_range", [1.0, 1.0])
            pore_canvas = self._apply_stretch(pore_canvas, stretch_range)

        if settings.get("rotation_enabled", False):
            angle = random.uniform(0, 360)
            pore_canvas = self._rotate_pore(pore_canvas, angle)

        return pore_canvas

    def _find_valid_placement(
        self, pore_canvas: np.ndarray, occupied_mask: np.ndarray, category: str
    ) -> tuple[int, int] | None:
        """Находит подходящие координаты для размещения поры."""
        margin_x = int(self._width * _IMAGE_MARGIN_FACTOR)
        margin_y = int(self._height * _IMAGE_MARGIN_FACTOR)
        canvas_center = pore_canvas.shape[0] // 2

        if (self._width - 2 * margin_x) <= 2 * canvas_center or (
            self._height - 2 * margin_y
        ) <= 2 * canvas_center:
            return None

        min_distance = self._get_min_distance_for_category(category)

        for _ in range(_PORE_PLACEMENT_ATTEMPTS):
            x = random.randint(
                margin_x + canvas_center, self._width - margin_x - canvas_center
            )
            y = random.randint(
                margin_y + canvas_center, self._height - margin_y - canvas_center
            )

            if self._can_place_pore(
                pore_canvas, occupied_mask, x, y, canvas_center, min_distance
            ):
                return x, y
        return None

    def _get_min_distance_for_category(self, category: str) -> int:
        """Возвращает минимально допустимое расстояние для категории пор."""
        if category == "large_pores":
            return _MIN_DISTANCE_LARGE
        if category == "medium_pores":
            return _MIN_DISTANCE_MEDIUM
        return _MIN_DISTANCE_SMALL

    def _apply_stretch(
        self, pore_canvas: np.ndarray, stretch_range: list[float]
    ) -> np.ndarray:
        """Растягивает холст с порой по случайной оси."""
        stretch_factor = random.uniform(*stretch_range)
        h, w = pore_canvas.shape

        if random.random() > 0.5:
            new_w, new_h = int(w * stretch_factor), h
        else:
            new_w, new_h = w, int(h * stretch_factor)

        stretched = cv2.resize(pore_canvas, (new_w, new_h), cv2.INTER_NEAREST)

        max_dim = max(new_h, new_w) + 4
        square_canvas = np.full((max_dim, max_dim), _WHITE_COLOR, np.uint8)

        y_off = (max_dim - new_h) // 2
        x_off = (max_dim - new_w) // 2

        square_canvas[y_off : y_off + new_h, x_off : x_off + new_w] = stretched

        return cv2.morphologyEx(
            square_canvas, cv2.MORPH_OPEN, _MORPH_KERNEL_ELLIPSE_3X3, iterations=1
        )

    def _rotate_pore(self, pore_canvas: np.ndarray, angle: float) -> np.ndarray:
        """Поворачивает холст с порой на заданный угол."""
        h, w = pore_canvas.shape
        center = (w // 2, h // 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            pore_canvas,
            rot_mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_WHITE_COLOR,
        )
        return cv2.morphologyEx(
            rotated, cv2.MORPH_OPEN, _MORPH_KERNEL_ELLIPSE_3X3, iterations=1
        )

    def _create_irregular_pore(
        self, radius: int, irregularity: float, spikiness: float, num_vertices: int
    ) -> np.ndarray:
        """Создает базовую неровную форму поры."""
        canvas_size = int(radius * 2.5)
        canvas = np.full((canvas_size, canvas_size), _WHITE_COLOR, np.uint8)
        center = canvas_size // 2

        angle_steps = [random.uniform(0.8, 1.2) for _ in range(num_vertices)]
        total_angle = sum(angle_steps)
        angle_steps = [s * 2 * np.pi / total_angle for s in angle_steps]

        points = []
        current_angle = 0
        for step in angle_steps:
            current_angle += step
            r_i = radius * (1 + random.uniform(-spikiness, spikiness))
            x = center + int(r_i * np.cos(current_angle))
            y = center + int(r_i * np.sin(current_angle))
            points.append([x, y])

        cv2.fillPoly(canvas, [np.array(points, dtype=np.int32)], _BLACK_COLOR)
        return cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, _MORPH_KERNEL_ELLIPSE_3X3)

    def _create_realistic_pore(
        self, radius: int, pore_type: str = "normal"
    ) -> np.ndarray:
        """Создает неидеальную, "реалистичную" пору заданного типа."""
        if pore_type == "smooth":
            return self._create_irregular_pore(
                radius,
                irregularity=random.uniform(0.1, 0.25),
                spikiness=random.uniform(0.05, 0.15),
                num_vertices=random.randint(12, 20),
            )
        if pore_type == "rough":
            return self._create_irregular_pore(
                radius,
                irregularity=random.uniform(0.25, 0.4),
                spikiness=random.uniform(0.15, 0.3),
                num_vertices=random.randint(8, 14),
            )
        return self._create_irregular_pore(
            radius,
            irregularity=random.uniform(0.15, 0.3),
            spikiness=random.uniform(0.1, 0.2),
            num_vertices=random.randint(10, 16),
        )

    def _can_place_pore(
        self,
        pore_canvas: np.ndarray,
        occupied_mask: np.ndarray,
        x: int,
        y: int,
        canvas_center: int,
        min_distance: int,
    ) -> bool:
        """Проверяет, можно ли разместить пору без коллизий."""
        y_start = y - canvas_center
        x_start = x - canvas_center
        slice_y, slice_x, pore_mask = self._get_overlap_slices(
            pore_canvas, y_start, x_start
        )

        occupied_region = occupied_mask[slice_y, slice_x]

        if min_distance > 0:
            kernel_size = 2 * min_distance + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            expanded_pore_mask = cv2.dilate(pore_mask.astype(np.uint8), kernel) > 0
        else:
            expanded_pore_mask = pore_mask

        return not np.any(occupied_region & expanded_pore_mask)

    def _place_on_image(
        self,
        image: np.ndarray,
        occupied_mask: np.ndarray,
        canvas: np.ndarray,
        x: int,
        y: int,
        size_category: str,
        type_category: str,
        original_radius: float,
    ) -> None:
        """Размещает холст с порой на изображении и обновляет маску."""
        canvas_center = canvas.shape[0] // 2
        y_start = y - canvas_center
        x_start = x - canvas_center

        slice_y, slice_x, pore_mask = self._get_overlap_slices(canvas, y_start, x_start)

        image[slice_y, slice_x][pore_mask] = _BLACK_COLOR
        occupied_mask[slice_y, slice_x][pore_mask] = True

        deformed_properties = self._calculate_pore_properties(canvas)

        original_properties = {
            "radius": original_radius,
            "area": np.pi * original_radius**2,
            "eccentricity": 0.0,  # Для круга всегда 0
            "circularity": 1.0,  # Для круга всегда 1
        }

        pore_info = {
            "center": (x, y),
            "original": original_properties,
            "deformed": deformed_properties,
            "size_category": size_category,
            "type_category": type_category,
            "canvas_shape": canvas.shape,
        }

        self._pore_data.append(pore_info)

    def _calculate_pore_properties(self, canvas: np.ndarray) -> dict[str, float]:
        """Вычисляет геометрические характеристики поры.

        Args:
            canvas: Холст с порой (бинарное изображение).

        Returns:
            Словарь с характеристиками поры:
            - radius: эквивалентный радиус
            - area: площадь в пикселях
            - eccentricity: эксцентриситет (0-1)
            - circularity: округлость (0-1)
        """
        # Создаем бинарную маску поры
        pore_mask = (canvas == _BLACK_COLOR).astype(np.uint8)

        # Находим контуры
        contours, _ = cv2.findContours(
            pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"radius": 0.0, "area": 0.0, "eccentricity": 0.0, "circularity": 0.0}

        # Берем самый большой контур (основная пора)
        largest_contour = max(contours, key=cv2.contourArea)

        # Вычисляем площадь
        area = cv2.contourArea(largest_contour)

        radius = np.sqrt(area / np.pi) if area > 0 else 0.0

        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (_, (width, height), _) = ellipse

            if width > 0 and height > 0:
                a = max(width, height) / 2
                b = min(width, height) / 2
                eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0.0
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter**2)
            circularity = min(1.0, circularity)  # Ограничиваем максимум 1.0
        else:
            circularity = 0.0

        return {
            "radius": radius,
            "area": area,
            "eccentricity": eccentricity,
            "circularity": circularity,
        }

    def _get_overlap_slices(
        self, canvas: np.ndarray, y_start: int, x_start: int
    ) -> tuple[slice, slice, np.ndarray]:
        """Вычисляет срезы для области пересечения поры и изображения."""
        h, w = canvas.shape

        img_y_start = max(0, y_start)
        img_y_end = min(self._height, y_start + h)
        img_x_start = max(0, x_start)
        img_x_end = min(self._width, x_start + w)

        canv_y_start = img_y_start - y_start
        canv_y_end = img_y_end - y_start
        canv_x_start = img_x_start - x_start
        canv_x_end = img_x_end - x_start

        pore_slice = canvas[canv_y_start:canv_y_end, canv_x_start:canv_x_end]
        pore_mask = pore_slice == _BLACK_COLOR

        return (
            slice(img_y_start, img_y_end),
            slice(img_x_start, img_x_end),
            pore_mask,
        )

    def get_pore_data(self) -> list[dict[str, Any]]:
        """Возвращает информацию о всех сгенерированных порах.

        Returns:
            Список словарей с информацией о каждой поре.
        """
        return self._pore_data.copy()

    def save_pore_data_to_json(self, filepath: str | Path) -> None:
        """Сохраняет информацию о порах в JSON файл.

        Args:
            filepath: Путь к файлу для сохранения данных.
        """
        json_data = []
        for pore in self._pore_data:
            pore_dict = {
                "center_x": int(pore["center"][0]),
                "center_y": int(pore["center"][1]),
                "radius": float(pore["radius"]),
                "area": float(pore["area"]),
                "eccentricity": float(pore["eccentricity"]),
                "circularity": float(pore["circularity"]),
                "size_category": pore["size"],
                "type_category": pore["type"],
                "canvas_width": int(pore["canvas_shape"][1]),
                "canvas_height": int(pore["canvas_shape"][0]),
            }
            json_data.append(pore_dict)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def get_current_pore_data(self) -> list[dict[str, Any]]:
        """Возвращает данные о порах для текущего изображения."""
        pore_data_formatted = []
        for pore in self._pore_data:
            pore_dict = {
                "center_x": int(pore["center"][0]),
                "center_y": int(pore["center"][1]),
                "original": {
                    "radius": round(float(pore["original"]["radius"]), 2),
                    "area": round(float(pore["original"]["area"]), 2),
                    "eccentricity": round(float(pore["original"]["eccentricity"]), 4),
                    "circularity": round(float(pore["original"]["circularity"]), 4),
                },
                "deformed": {
                    "radius": round(float(pore["deformed"]["radius"]), 2),
                    "area": round(float(pore["deformed"]["area"]), 2),
                    "eccentricity": round(float(pore["deformed"]["eccentricity"]), 4),
                    "circularity": round(float(pore["deformed"]["circularity"]), 4),
                },
                "size_category": pore["size_category"],
                "type_category": pore["type_category"],
                "canvas_width": int(pore["canvas_shape"][1]),
                "canvas_height": int(pore["canvas_shape"][0]),
            }
            pore_data_formatted.append(pore_dict)
        return pore_data_formatted
