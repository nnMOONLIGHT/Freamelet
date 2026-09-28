# -*- coding: utf-8 -*-
"""
Style catalogue for the visual-style classifier.

Each entry describes one aesthetic cluster the network can recognise from
the low-level visual signature of a photo (colour, contrast, saturation,
edge density, symmetry). The palette/clothing/hobby fields are product
copy shown to the user once the network has picked a class - they carry
no weight during training.
"""

STYLES = [
    {
        "id": "minimal",
        "name": "Минимализм",
        "summary": "Чистые линии, простор и спокойная палитра.",
        "palette": ["#f5f3fb", "#c9c9d6", "#7a6df0", "#14162c"],
        "clothing": ["Однотонная база", "Прямой крой", "Один акцентный предмет", "Простая обувь без деталей"],
        "hobbies": ["Дизайн интерьеров", "Графический дизайн", "Фотография предметки"],
        "directions": ["design", "3d"],
    },
    {
        "id": "street",
        "name": "Стрит-стайл",
        "summary": "Крупный крой, городская фактура, уверенный контраст.",
        "palette": ["#12142c", "#3a8bef", "#e0567c", "#f5f3fb"],
        "clothing": ["Оверсайз худи", "Карго-брюки", "Кроссовки", "Бейсболка"],
        "hobbies": ["Стрит-арт", "Скейт-фотография", "Дизайн принтов"],
        "directions": ["fashion", "media"],
    },
    {
        "id": "retro",
        "name": "Ретро / Винтаж",
        "summary": "Тёплые приглушённые тона и налёт старой плёнки.",
        "palette": ["#dc9a2e", "#8a5a3a", "#f0e0c0", "#4a3527"],
        "clothing": ["Вельвет", "Высокая талия", "Платок или шарф", "Круглые очки"],
        "hobbies": ["Плёночная фотография", "Иллюстрация", "Керамика"],
        "directions": ["craft", "media"],
    },
    {
        "id": "romantic",
        "name": "Романтика / Пастель",
        "summary": "Мягкий свет, воздушные пастельные оттенки.",
        "palette": ["#fce6ec", "#e2dcfc", "#efebfe", "#c98aa0"],
        "clothing": ["Струящиеся ткани", "Пастельные тона", "Мелкие аксессуары", "Балетки"],
        "hobbies": ["Иллюстрация", "Флористика", "Леттеринг"],
        "directions": ["craft", "text"],
    },
    {
        "id": "grunge",
        "name": "Панк / Гранж",
        "summary": "Резкий контраст, потёртая фактура, характер.",
        "palette": ["#0c0d20", "#8f83ff", "#e0567c", "#9598b8"],
        "clothing": ["Косуха", "Рваный деним", "Массивная обувь", "Металлическая фурнитура"],
        "hobbies": ["Музыка и биты", "Коллаж", "Тату-эскизы"],
        "directions": ["media", "text"],
    },
    {
        "id": "avant",
        "name": "Авангард / Эксперимент",
        "summary": "Смелые формы, насыщенный цвет, нестандартная композиция.",
        "palette": ["#7a6df0", "#dc9a2e", "#26a677", "#e0567c"],
        "clothing": ["Асимметричный крой", "Смешение фактур", "Необычные силуэты", "Скульптурные аксессуары"],
        "hobbies": ["3D-моделирование", "Digital-арт", "Экспериментальная мода"],
        "directions": ["3d", "fashion"],
    },
    {
        "id": "classic",
        "name": "Классика / Элегантность",
        "summary": "Сбалансированные пропорции и сдержанная палитра.",
        "palette": ["#14162c", "#63678a", "#e7e5f3", "#7a6df0"],
        "clothing": ["Пиджак прямого кроя", "Нейтральные тона", "Лаконичные аксессуары", "Классическая обувь"],
        "hobbies": ["Портретная фотография", "Каллиграфия", "Дизайн одежды"],
        "directions": ["fashion", "design"],
    },
    {
        "id": "popart",
        "name": "Поп-арт / Яркий",
        "summary": "Максимум цвета, смелые контрасты, энергия.",
        "palette": ["#e0567c", "#3a8bef", "#f6c667", "#26a677"],
        "clothing": ["Яркие цветовые блоки", "Принты", "Контрастные сочетания", "Акцентные украшения"],
        "hobbies": ["Иллюстрация", "Анимация", "Дизайн мерча"],
        "directions": ["design", "media"],
    },
]

STYLE_INDEX = {s["id"]: i for i, s in enumerate(STYLES)}
NUM_CLASSES = len(STYLES)

FEATURE_NAMES = [
    "r_mean", "g_mean", "b_mean",
    "h_mean", "s_mean", "v_mean",
    "brightness_std", "saturation_std",
    "warm_cool", "colorfulness",
    "edge_density", "edge_std",
    "dark_ratio", "light_ratio",
    "symmetry",
]
NUM_FEATURES = len(FEATURE_NAMES)
