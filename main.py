"""Основной модуль для генерации набора данных изображений пор."""

import json
import os
import random
import string
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from src.config_loader import ConfigLoader
from src.image_processor import ImageProcessor
from src.pore_generator import PoreGenerator

_DEFAULT_CONFIG_PATH = "config.json"


def _generate_run_id(length: int = 6) -> str:
    """Генерирует случайный буквенно-цифровой ID для сессии генерации."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


class PoreImageGenerator:
    """Оркестрирует процесс создания датасета изображений пор.

    Загружает конфигурацию, инициализирует компоненты и генерирует пары
    изображений: чистое (маска) и финальное зашумленное.

    Attributes:
        config: Словарь с полной конфигурацией.
        pore_generator: Экземпляр генератора базовых изображений пор.
        image_processor: Экземпляр обработчика изображений.
        _clean_dir: Путь к директории для чистых изображений.
        _noisy_dir: Путь к директории для зашумленных изображений.
        _total_images: Общее количество изображений для генерации.
        _all_pores_data: Словарь для хранения данных о порах всех изображений.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH, verbose: bool = False):
        """Инициализирует генератор.

        Args:
            config_path: Путь к файлу конфигурации.
            verbose: Флаг для вывода подробной информации.
        """
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        self.verbose = verbose

        self.pore_generator = PoreGenerator(self.config)
        self.image_processor = ImageProcessor(self.config)

        output_settings = self.config.get("output_settings", {})
        image_settings = self.config.get("image_settings", {})

        self._clean_dir = output_settings.get("clean_dir", "output/clean")
        self._noisy_dir = output_settings.get("noisy_dir", "output/noisy")
        self._total_images = image_settings.get("total_images", 100)

        # Словарь для хранения данных о порах всех изображений
        self._all_pores_data = {}

        self._create_output_directories()

    def _create_output_directories(self) -> None:
        """Создает выходные директории, если они не существуют."""
        os.makedirs(self._clean_dir, exist_ok=True)
        os.makedirs(self._noisy_dir, exist_ok=True)

    def generate_images(self) -> None:
        """Запускает основной цикл генерации и сохранения изображений."""
        run_id = _generate_run_id()
        print(
            f"Starting generation of {self._total_images} image pairs "
            f"(Run ID: {run_id})..."
        )

        for i in tqdm(range(self._total_images), desc="Generating Images"):
            self._generate_and_save_pair(run_id, i)

        self._save_all_pores_data()

    def _generate_and_save_pair(self, run_id: str, image_index: int) -> None:
        """Генерирует, обрабатывает и сохраняет одну пару изображений."""
        clean_image = self.pore_generator.generate_image()

        if self.verbose:
            print(f"Сгенерировано пор: {len(self.pore_generator._pore_data)}")

        current_pore_data = self.pore_generator.get_current_pore_data()

        noisy_image = self.image_processor.add_complete_noise(clean_image.copy())

        clean_image_final = self.image_processor.crop(clean_image)
        noisy_image_final = self.image_processor.crop(noisy_image)

        base_filename = f"{run_id}_{image_index:04d}"
        clean_filename = f"{base_filename}_clean.png"
        noisy_filename = f"{base_filename}_noisy.png"

        clean_path = os.path.join(self._clean_dir, clean_filename)
        noisy_path = os.path.join(self._noisy_dir, noisy_filename)

        cv2.imwrite(clean_path, clean_image_final)
        cv2.imwrite(noisy_path, noisy_image_final)

        self._all_pores_data[clean_filename] = current_pore_data

    def _save_all_pores_data(self) -> None:
        """Сохраняет данные о порах всех изображений в JSON файл."""
        output_path = Path("output/pores_data.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._all_pores_data, f, indent=2, ensure_ascii=False)

        print(f"Pore data saved to {output_path}")
        print(f"Total images processed: {len(self._all_pores_data)}")
        total_pores = sum(len(pores) for pores in self._all_pores_data.values())
        print(f"Total pores generated: {total_pores}")


@click.command()
@click.option("--verbose", is_flag=True)
def main(verbose) -> None:
    """Основная точка входа для запуска генератора изображений."""
    generator = PoreImageGenerator(_DEFAULT_CONFIG_PATH, verbose)
    generator.generate_images()


if __name__ == "__main__":
    main()
