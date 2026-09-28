"""Lexical building blocks of agent_1: text normalisation, analyzers, the item-params parser, field BM25.

Everything here is pure and deterministic, so the orchestrator can promote it into ``src/`` as is.

Params format. ``item_infm_params_text`` (and the query filter ``search_infm_params_text``) is one glued
string of ``key value key value ...`` pairs, e.g.

    Вид услуги Красота, здоровье Тип услуги Маникюр, педикюр Место оказания услуг Москва, ул. ... Начальная
    цена Тип стоимости за услугу Услуга Аппаратный маникюр Стоимость 1250 Продолжительность 1 ч ...

Keys come from a fixed schema (Avito parameter names); values are either closed-vocabulary phrases
("Красота, здоровье"), numbers, or free text (service names, addresses). The key vocabulary below was mined
from the corpus (``experiments/agent_1/e1_params.py`` prints the statistics): split the string into
segments at capitalised tokens and keep frequent segments that are preceded by many different segments
(a value is almost always preceded by its own key, a key by many different values), then curated by hand
(street and person names inside addresses pass the statistical test and were removed).

Parsing = greedy longest match of known keys at capitalised tokens. Keys of kind ``flag`` never carry a
value ("Предоплата", "Гарантия", ...); to avoid cutting free-text values that merely start with such a word
("Услуга Выезд мастера"), a flag is accepted only when another key (or the end of the string) follows it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

# ----------------------------------------------------------------------------------------------------
# Normalisation and analyzers
# ----------------------------------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


def normalize(text: str | None) -> str:
    """Lower case and ё→е (the only normalisation shared by every analyzer)."""
    return (text or "").lower().replace("ё", "е")


def tokens(text: str | None) -> list[str]:
    """Alphanumeric tokens of the normalised text (hyphenated words are split: гель-лак -> гель, лак)."""
    return TOKEN_RE.findall(normalize(text))


_STEM_CACHE: dict[str, str] = {}
_LEMMA_CACHE: dict[str, str] = {}
_stemmer = None
_morph = None


def stem(word: str) -> str:
    global _stemmer
    s = _STEM_CACHE.get(word)
    if s is None:
        if _stemmer is None:
            import snowballstemmer
            _stemmer = snowballstemmer.stemmer("russian")
        s = _STEM_CACHE[word] = _stemmer.stemWord(word)
    return s


def lemma(word: str) -> str:
    """pymorphy3 normal form of the most probable parse (latin words and numbers are kept as is)."""
    global _morph
    s = _LEMMA_CACHE.get(word)
    if s is None:
        if not re.search(r"[а-я]", word):
            s = word
        else:
            if _morph is None:
                import pymorphy3
                _morph = pymorphy3.MorphAnalyzer()
            s = _morph.parse(word)[0].normal_form.replace("ё", "е")
        _LEMMA_CACHE[word] = s
    return s


def analyze_stem(text: str | None) -> list[str]:
    return [stem(w) for w in tokens(text)]


def analyze_lemma(text: str | None) -> list[str]:
    return [lemma(w) for w in tokens(text)]


def analyze_both(text: str | None) -> list[str]:
    """Stems and lemmas as two token streams in one bag (prefixed so that they never collide)."""
    out = []
    for w in tokens(text):
        out.append("s:" + stem(w))
        out.append("l:" + lemma(w))
    return out


ANALYZERS = {"stem": analyze_stem, "lemma": analyze_lemma, "both": analyze_both}

# ----------------------------------------------------------------------------------------------------
# Params parser
# ----------------------------------------------------------------------------------------------------

# kind -> keys. Kinds: vid / tip / tip_auto / service / service_name / place (fields used directly);
# where / who / clients / experience (structured facts); desc (descriptive enumerations: what exactly the
# service is - they go to the "other params" text field); money / schedule / misc (boilerplate);
# flag (no value).
KEYS: dict[str, list[str]] = {
    "vid": ["Вид услуги"],
    "tip": ["Тип услуги"],
    "tip_auto": ["Тип услуги автосервиса"],
    "service": ["Услуга", "Услуги"],
    "service_name": ["Название услуги"],
    "place": ["Место оказания услуг", "Место сделки", "Место проживания", "Адрес"],
    "where": ["Где вы оказываете услуги", "Как вы работаете", "Как работаете", "Куда выезжаете", "Формат",
              "Где проводите занятия", "Где снимаете", "Выезжаете ли к заказчику"],
    "who": ["Кто оказывает услуги", "Преподаватель"],
    "clients": ["Ваши клиенты", "Для кого", "Аудитория", "Кому подойдёт место"],
    "experience": ["Опыт работы"],
    "desc": [
        "Чем вы занимаетесь", "Предмет или специальность", "Специальность", "Специальность или сфера",
        "Специализация", "Направление", "Мероприятия", "Марка авто", "Марка", "Модель", "Производители",
        "Тип техники", "Тип обслуживаемой техники", "Тип автосервиса", "Тип оборудования", "Груз", "Перевозка",
        "Вид транспорта", "Транспорт", "Тип транспорта", "Тип грузового транспорта", "Тип помещения",
        "Какими комнатами занимаетесь", "Какими нежилыми помещениями занимаетесь",
        "Какими жилыми помещениями занимаетесь", "Что декорируете", "Что перевозите", "Занятия", "Уровень",
        "Язык перевода", "Бытовая техника", "Мультимедиа", "Коммуникации", "Стиль танца",
        "Жанр или формат", "Тип товара", "Вид товара", "Вид объекта", "Специфика", "Дополнительно",
        "Кузов", "Поездка", "Тип подъёмной техники", "Дополнительное оборудование", "Тип двигателя",
        "Тип кузова", "Класс авто", "Тип аренды", "Способ погрузки", "Музыкальный инструмент", "Профессия",
        "Профессия в опыте работы", "Должность", "Группа профессий", "Бизнес-категория", "Серия",
        "Коробка передач", "Привод", "Создание образа", "Анимация", "Средства для уборки", "Работа со ссылками",
        "С какими", "Техника", "Машины", "Загрузка", "Аренда", "В каких стилях работаете", "Телевизоры",
    ],
    "money": [
        "Начальная цена", "Тип стоимости", "Стоимость", "Продолжительность", "Цена с", "Цена, ₽",
        "Выполняю заказы от", "Минимальная сумма заказа", "Минимальное время заказа", "Минимальное время аренды",
        "Минимальное количество суток", "Минимальный бюджет клиента", "Сумма за весь срок аренды", "Депозит",
        "Способ оплаты", "Как внести", "Доплата за километр сверх лимита", "Доплата за бензин",
        "Доплата за выдачу/получение в нерабочее время", "Скидка за количество", "Страховка",
    ],
    "schedule": [
        "График работы от", "График работы до", "График работы, дни недели", "График работы", "Время работы, с",
        "Время работы, до", "Время для связи от", "Время для связи до", "Время для связи, дни недели",
        "Время для связи, с", "Время для связи, до", "Рабочие дни", "Дни", "Начало работы", "Курс длится",
    ],
    "misc": [
        "Признак предзаполнения прайс листа", "Признак мигрированного прайс листа", "Бригада", "Гости",
        "Исполнителей в команде", "Сколько человек может участвовать", "Сколько гостей готовы снимать",
        "Число исполнителей", "Число мест", "Грузоподъёмность", "Грузоподъёмность до", "Высота груза",
        "Длина груза", "Ширина груза", "Грузчики", "Расстояние доставки от", "Расстояние доставки до",
        "Площадь объекта от", "Год окончания", "Учебное учреждение", "Название компании", "Название заведения",
        "Год выпуска", "Количество дверей", "Поколение", "Модификация", "Комплектация", "Руль",
        "Государственный номер", "Минимальный возраст водителя", "Минимальный стаж вождения",
        "Сколько километров в сутки", "Через сколько дней возвращается", "Длина стрелы",
        "Объём копательного ковша", "Тип занятости", "Бонусы от работодателя", "Этаж", "Количество комнат",
        "Общая площадь", "Расстояние до города", "Цвет", "Дата рождения", "Образование", "Срок аренды",
        "Система", "Районы", "Метро", "Видеофайлы", "Провайдер", "Состояние", "Вид объявления",
        "Камера наблюдения в ремонтной зоне", "Можно со своими запчастями", "Период для детского кресла",
        "Дополнительная педаль", "Рейтинг пользователя", "Инструменты", "Проживание на объекте",
        "Сертификат после обучения", "Трудоустройство после обучения", "Оплата в рассрочку",
        "Срочная услуга (мультистатус)", "Слова в описании",
    ],
    "flag": [
        "Работа по договору", "Гарантия", "Гарантия на работу", "Гарантия на выполнение работ",
        "Готов закупить материалы", "Предоплата", "Берёте ли срочные заказы", "Бесплатная консультация",
        "Работаете с юрлицами и ИП", "Работаете с НДС", "Онлайн-показ", "Онлайн-запись", "Есть портфолио",
        "Работа с техникой премиум-класса", "Выезд в день заказа", "Удалённые консультации",
        "Своя музыкальная аппаратура", "Готовность к командировкам", "Закупка материалов",
        "Преподаёт носитель языка", "Подбираете ли студию", "Безопасный показ", "Онлайн-бронирование",
        "С оператором", "Выезд за город", "Выезд к клиенту", "Выезд", "Тренирует мастер спорта",
        "Есть кран-манипулятор", "Детское кресло",
        "Есть лимит пробега", "Работа в праздники и выходные", "Залог", "Применен автозаголовок",
        "Поэтапная оплата", "Доставка", "Составление сценария", "Техническая поддержка сайта", "Печать фотокниги",
        "Участие в пилоте CPL",
    ],
}
KEY_KIND: dict[str, str] = {k: kind for kind, keys in KEYS.items() for k in keys}
_KEY_TOKENS = sorted(({tuple(k.split(" ")) for k in KEY_KIND}), key=len, reverse=True)
_BY_FIRST: dict[str, list[tuple[str, ...]]] = defaultdict(list)
for _kt in _KEY_TOKENS:
    _BY_FIRST[_kt[0]].append(_kt)          # longest first (the list is sorted by length)


def _match_any(toks: list[str], i: int) -> tuple[str, ...] | None:
    """Longest key whose tokens start at position i (no flag condition)."""
    for kt in _BY_FIRST.get(toks[i], ()):
        if tuple(toks[i:i + len(kt)]) == kt:
            return kt
    return None


_PRICE_CONTEXT = {"money", "service", "service_name"}
# One-word generic keys ("Техника", "Аренда", "Груз", ...) also start many values ("Груз Техника и оборудование",
# "Название услуги Аренда генератора"): they may not open a new field while a value-carrying key is still empty.
_WEAK_KINDS = {"desc", "where", "who", "clients", "flag", "misc"}
_BLOCKING_KINDS = {"vid", "tip", "tip_auto", "service", "service_name", "place", "where", "who", "clients",
                   "experience", "desc"}


def parse_params(text: str | None) -> list[tuple[str, str]]:
    """Split a glued params string into an ordered list of (key, value); value may be ''.

    Tokens before the first recognised key are returned under the key ''.
    """
    toks = [t for t in (text or "").split(" ") if t]
    n = len(toks)
    out: list[tuple[str, str]] = []
    cur_key, cur_val = "", []
    i = 0
    while i < n:
        kt = None
        if toks[i][:1].isupper():
            for cand in _BY_FIRST.get(toks[i], ()):
                if tuple(toks[i:i + len(cand)]) != cand:
                    continue
                j = i + len(cand)
                kind = KEY_KIND[" ".join(cand)]
                if kind == "flag" and j < n and _match_any(toks, j) is None:
                    continue                      # a flag must be followed by a key (flags carry no value)
                if kind == "service" and KEY_KIND.get(cur_key) not in _PRICE_CONTEXT:
                    continue                      # "Тип услуги Услуги парикмахера": a value, not a price-list entry
                if (len(cand) == 1 and kind in _WEAK_KINDS and not cur_val
                        and KEY_KIND.get(cur_key) in _BLOCKING_KINDS):
                    continue
                kt = cand
                break
        if kt is not None:
            if cur_key or cur_val:
                out.append((cur_key, " ".join(cur_val)))
            cur_key, cur_val = " ".join(kt), []
            i += len(kt)
        else:
            cur_val.append(toks[i])
            i += 1
    if cur_key or cur_val:
        out.append((cur_key, " ".join(cur_val)))
    return out


def params_dict(text: str | None) -> dict[str, list[str]]:
    """key -> list of values (in order of appearance, empty values included)."""
    d: dict[str, list[str]] = defaultdict(list)
    for k, v in parse_params(text):
        d[k].append(v)
    return d


_REGION_RE = re.compile(r"(област|обл\.|край|республик|автономн|округ|район|волость|поселение|сельсовет|"
                        r"товарищество|муниципальн)", re.I)
_STREET_RE = re.compile(r"(^|\s)(ул\.|улица|пр-т|проспект|пер\.|переулок|ш\.|шоссе|пл\.|площадь|б-р|бульвар|наб\.|"
                        r"набережная|проезд|мкр|микрорайон|тракт|линия|метро|жилой комплекс|кв-л|квартал|тупик|"
                        r"аллея|снт|днп|территория|посёлок|поселок|пос\.|деревня|село|станица|\d)", re.I)


def place_city(place: str) -> str:
    """Best guess of the settlement in an address ("Московская обл., Щёлково, ул. ..." -> "Щёлково")."""
    parts = [p.strip() for p in place.split(",") if p.strip()]
    for p in parts:
        if _REGION_RE.search(p) or _STREET_RE.search(p):
            continue
        return p
    return ""


_ONLINE = {"Удалённо", "Онлайн", "Онлайн с преподавателем", "Удалённые консультации"}
_HOME = {"У себя", "В салоне", "В мастерской", "У себя дома", "В студии", "Сервисный центр"}
_VISIT = {"У клиента", "У заказчика дома", "Выезд к клиенту", "По всему городу", "В выбранные зоны",
          "Выезд за город", "По городу"}


def _experience_years(vals: list[str]) -> float | None:
    for v in vals:
        v = v.strip()
        if not v:
            continue
        if v.startswith("Меньше"):
            return 0.5
        if (m := re.match(r"(\d+)\s*[–-]\s*(\d+)", v)):
            return (int(m.group(1)) + int(m.group(2))) / 2
        if (m := re.match(r"(\d+)", v)):
            return float(m.group(1))
    return None


def item_params_record(text: str | None) -> dict:
    """Structured record of one item's params (the row of table agent_1/items_params)."""
    d = params_dict(text)
    first = lambda k: next((v for v in d.get(k, []) if v), "")     # noqa: E731
    services = [v for v in d.get("Услуга", []) + d.get("Услуги", []) if v and v != "Своя услуга"]
    custom = [v for v in d.get("Название услуги", []) if v]
    where_vals = {v for k in KEYS["where"] for v in d.get(k, [])}
    flags = {k for k in d if KEY_KIND.get(k) == "flag"}
    desc_vals = [v for k in KEYS["desc"] for v in d.get(k, []) if v]
    prices = []
    for v in d.get("Стоимость", []):
        m = re.match(r"(\d+)", v)
        if m:
            prices.append(int(m.group(1)))
    place = first("Место оказания услуг") or first("Место сделки")
    return {
        "vid": first("Вид услуги"),
        "tip": first("Тип услуги"),
        "tip_auto": first("Тип услуги автосервиса"),
        "services": list(dict.fromkeys(services)),
        "custom_services": list(dict.fromkeys(custom)),
        "place": place,
        "place_city": place_city(place),
        "where_online": bool(where_vals & _ONLINE) or "Удалённые консультации" in flags,
        "where_home": bool(where_vals & _HOME),
        "where_visit": bool(where_vals & _VISIT) or "Выезд к клиенту" in flags,
        "no_visit": "Не выезжаю" in where_vals,
        "who": first("Кто оказывает услуги") or first("Преподаватель"),
        "clients": sorted({v for k in KEYS["clients"] for v in d.get(k, []) if v}),
        "experience_years": _experience_years(d.get("Опыт работы", [])),
        "n_services": len(services) + len(custom),
        "n_prices": len(prices),
        "min_list_price": min((p for p in prices if p > 0), default=None),
        "has_price": any(p > 0 for p in prices),
        "desc_values": list(dict.fromkeys(desc_vals)),
        "unparsed_head": d.get("", [""])[0] if "" in d else "",
        "n_keys": len(d),
    }


def core_text(rec: dict) -> str:
    """What the item *is*: Вид + Тип (+ auto) + price-list service names + custom service names."""
    return " . ".join([rec["vid"], rec["tip"], rec["tip_auto"], *rec["services"], *rec["custom_services"]])


def other_params_text(rec: dict) -> str:
    """Descriptive enumerations (specialities, brands, events, cargo types, ...) — no boilerplate."""
    return " . ".join(rec["desc_values"])


# ----------------------------------------------------------------------------------------------------
# Query filters
# ----------------------------------------------------------------------------------------------------

def parse_filters(text: str | None) -> dict[str, list[str]]:
    """Query filter text -> {key: [values]} with empty values dropped ("Вид услуги" alone = no filter)."""
    d: dict[str, list[str]] = {}
    for k, v in parse_params(text):
        if k and v:
            d.setdefault(k, []).append(v)
        elif k and KEY_KIND.get(k) == "flag":
            d.setdefault(k, [])
    return d


# ----------------------------------------------------------------------------------------------------
# BM25 over one field (sparse algebra)
# ----------------------------------------------------------------------------------------------------

class FieldBM25:
    """BM25 of one text field. ``scores(queries)`` = binary(query terms) @ W^T, W = per-(doc, term) weights.

    ``vocab`` maps analyzed terms to columns. The document-term count matrix is kept so that k1/b can be
    re-tuned without re-tokenising (``reweight``).
    """

    def __init__(self, docs_terms: list[list[str]], k1: float = 1.2, b: float = 0.75,
                 vocab: dict[str, int] | None = None):
        if vocab is None:
            vocab = {}
            for terms in docs_terms:
                for t in terms:
                    if t not in vocab:
                        vocab[t] = len(vocab)
        self.vocab = vocab
        rows, cols = [], []
        for r, terms in enumerate(docs_terms):
            for t in terms:
                c = vocab.get(t)
                if c is not None:
                    rows.append(r)
                    cols.append(c)
        tf = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (np.array(rows, dtype=np.int32),
                            np.array(cols, dtype=np.int32))), shape=(len(docs_terms), len(vocab)))
        tf.sum_duplicates()
        self.tf = tf
        self.reweight(k1, b)

    def reweight(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        tf = self.tf.tocoo()
        n_docs = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(self.tf.sum(axis=1)).ravel()
        avg = dl.mean() if dl.mean() > 0 else 1.0
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = sp.csr_matrix((tf.data * (k1 + 1) / denom * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self.wt = w.T.tocsr().astype(np.float32)

    def query_matrix(self, queries_terms: list[list[str]]) -> sp.csr_matrix:
        rows, cols = [], []
        for r, terms in enumerate(queries_terms):
            for t in set(terms):
                c = self.vocab.get(t)
                if c is not None:
                    rows.append(r)
                    cols.append(c)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(queries_terms), len(self.vocab)))

    def scores(self, queries_terms: list[list[str]]) -> np.ndarray:
        """Dense (n_queries, n_docs) float32 BM25 scores."""
        return (self.query_matrix(queries_terms) @ self.wt).toarray()
