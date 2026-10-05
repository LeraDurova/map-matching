import xml.etree.ElementTree as ET
import math
import heapq
from functools import lru_cache
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pyproj import Transformer

GPS_LAT_LON = [
    (55.755540, 37.642800),
    (55.755690, 37.643300),
    (55.755820, 37.643850),
    (55.756050, 37.643890),
    (55.756380, 37.643790),
    (55.756760, 37.643700),
    (55.756910, 37.644100),
    (55.757020, 37.644750),
    (55.757300, 37.645340),
    (55.757610, 37.645850),
]

# GPS_LAT_LON = [
#     (55.759670, 37.631620),
#     (55.759440, 37.631680),
#     (55.759220, 37.631840),
#     (55.759010, 37.632100),
#     (55.758800, 37.632340),
#     (55.758570, 37.632530),
#     (55.758300, 37.632900),
#     (55.758070, 37.633130),
#     (55.757840, 37.633410),
#     (55.757600, 37.633650),
# ]


MAP_FILE = Path(__file__).with_name("map.osm.txt")
SEARCH_RADIUS_M = 50.0
GPS_SIGMA_M = 20.0
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)

#парсим кусок карты.
def parse_osm(filename):
    root = ET.parse(filename).getroot()
    #нужные типы дорог
    road_types = {"primary", "secondary", "tertiary", "unclassified", "service"}
    ways = {}
    needed = set() #мно-во айди точек
    #перебираем линии карты, среди них отбираем дороги
    for way in root.findall("way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")} #извлекаем хар-ки дороги
        #пропускаем ненужные типы и площади
        if tags.get("highway") not in road_types or tags.get("area") == "yes":
            continue
        ids = [int(nd.get("ref")) for nd in way.findall("nd")] #читаем айди точек пренадлежащих дороге
        for key in ("name", "highway", "oneway", "maxspeed"): #ключевые хар-ки дороги (название, тип, односторонность, огр-е скорости)
            tags.setdefault(key, None)
        ways[int(way.get("id"))] = {"node_ids": ids, "tags": tags}
        needed.update(ids) #айди точкек дороги закидываем в множество

    points_xy = {} #тут будут координаты точек в метрах
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        #берем только точки, принадлежащие выбранным дорогам
        if node_id in needed:
            points_xy[node_id] = TRANSFORMER.transform(
                float(node.get("lon")), float(node.get("lat")), errcheck=True)
    if needed - points_xy.keys(): #проверка на потерю точек
        raise ValueError(f"В карте отсутствуют точки: {sorted(needed - points_xy.keys())}")

    #тут будем делить дороги на сегменты, содерж-е 2 точки (нач и кон)
    segments = {}
    for way_id, way in ways.items():
        tags, ids = way["tags"], way["node_ids"]
        #смотрим, разрешено ли движение только в одну сторону
        oneway = tags["oneway"]
        if oneway not in (None, "", "yes", "1", "true", "-1", "no", "0", "false"):
            raise ValueError(f"Не поддерживается oneway")
        if oneway in (None, "") and tags.get("junction") == "roundabout":
            oneway = "yes"
        forward = oneway != "-1"
        backward = oneway not in ("yes", "1", "true")
        #перебираем соседние точки
        for i, (start, end) in enumerate(zip(ids, ids[1:])):
            x0, y0 = points_xy[start]
            x1, y1 = points_xy[end]
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 and dy == 0:
                continue
            segments[(way_id, i)] = {
                "start_node": start, "end_node": end,
                "x0": x0, "y0": y0, "dx": dx, "dy": dy,
                "length_m": math.hypot(dx, dy),
                "forward": forward, "backward": backward, "tags": tags,
            }
    if not segments:
        raise ValueError("!!В карте нет подходящих дорожных сегментов!!")
    return points_xy, segments


#координаты любого положения на сегменте: t=0 начало, t=1 конец
def point_on_segment(segment, t):
    if 0 <= t <= 1:
        return segment["x0"] + t * segment["dx"], segment["y0"] + t * segment["dy"]
    else:
        raise ValueError("Параметр t вышел за границы отрезка !!! 0 <= t <= 1 !!!")

#по сути геометрия. проецируем точку на сегменты -> кандидаты
def get_candidates(gps_nodes, segments, radius, sigma):
    if not all(math.isfinite(v) and v > 0 for v in (radius, sigma)):
        raise ValueError("!!Радиус и sigma должны быть положительными конечными числами!!")
    #слои кандидатов
    layers = []
    for gps in gps_nodes:
        if not all(math.isfinite(gps[k]) for k in ("x", "y")):
            raise ValueError("!!Координаты GPS должны быть конечными числами!!")
        #кандидаты текущей GPS-точки
        candidates = []
        for sid, segment in segments.items():
            dx, dy = segment["dx"], segment["dy"]
            #находим параметр проекции GPS-точки на прямую сегмента
            t = ((gps["x"] - segment["x0"]) * dx
                 + (gps["y"] - segment["y0"]) * dy) / (dx * dx + dy * dy)
            #если проекция за пределами отрезка, берем ближайший конец
            t = max(0.0, min(1.0, t))
            x, y = point_on_segment(segment, t)
            #расстояние от GPS-точки до кандидата
            distance = math.hypot(gps["x"] - x, gps["y"] - y)
            #оставляем кандидата, только если он попал в радиус поиска
            if distance <= radius:
                candidates.append({"segment_id": sid, "t": t, "x": x, "y": y,
                                   "distance_m": distance,
                                   "log_observation": -0.5 * (distance / sigma) ** 2}) #логарифм оценки близости N; общий постоянный множитель опущен.
        layers.append(candidates)
    return layers

#строим дорожный граф. точки OSM + кандидаты, ребра это части дорог.
def build_road_graph(segments, layers):
    #для каждого сегмента собираем положения разбиения. концы уже включены
    positions = {sid: {0.0, 1.0} for sid in segments}
    for layer in layers:
        for candidate in layer:
            #добавляем положение кандидата.
            positions[candidate["segment_id"]].add(candidate["t"])

    #получаем ключ вершины графа по сегменту и положению на нем
    def vertex(sid, t):
        if t == 0:
            return ("node", segments[sid]["start_node"])
        if t == 1:
            return ("node", segments[sid]["end_node"])
        return ("segment", sid, t)

    #ключ это вершина. значение это список ребер выходящих их вершины
    graph = {}
    for sid, ts in positions.items():
        segment = segments[sid]
        ts = sorted(ts)
        for t0, t1 in zip(ts, ts[1:]):
            #ключи начала и конца части сегмента
            u, v = vertex(sid, t0), vertex(sid, t1)
            #создаем списки ребер для новых вершин
            graph.setdefault(u, [])
            graph.setdefault(v, [])
            length = (t1 - t0) * segment["length_m"]
            if segment["forward"]:
                graph[u].append({"to": v, "length_m": length,
                                 "segment_id": sid, "t0": t0, "t1": t1})
            if segment["backward"]:
                graph[v].append({"to": u, "length_m": length,
                                 "segment_id": sid, "t0": t1, "t1": t0})
    for layer in layers:
        for candidate in layer:
            candidate["vertex"] = vertex(candidate["segment_id"], candidate["t"])
    return graph

#ищем кратчайшие расстояния
def dijkstra(graph, source):
    distances, previous = {source: 0.0}, {}
    queue = [(0.0, source)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for edge in graph[vertex]:
            target = edge["to"]
            alternative = distance + edge["length_m"]
            if alternative < distances.get(target, math.inf):
                distances[target] = alternative
                previous[target] = (vertex, edge)
                heapq.heappush(queue, (alternative, target))
    return distances, previous

#оцениваем вероятность перехода.
def transition_score(gps_a, gps_b, a, b, road_distance):
    if not math.isfinite(road_distance):
        return -math.inf
    gx, gy = gps_b["x"] - gps_a["x"], gps_b["y"] - gps_a["y"]
    direct = math.hypot(gx, gy)
    maximum = max(direct, road_distance)
    transmission = min(direct, road_distance) / maximum if maximum else 1.0
    cx, cy = b["x"] - a["x"], b["y"] - a["y"]
    chord = math.hypot(cx, cy)
    direction = 1.0
    if direct > 0 and chord > 0:
        cosine = (gx * cx + gy * cy) / (direct * chord)
        direction = 1 - math.acos(max(-1.0, min(1.0, cosine))) / math.pi
    if transmission <= 0 or direction <= 0:
        return -math.inf
    return math.log(transmission) + math.log(direction)


#выбор посл-ти кандидатов
def match_gps(gps_nodes, segments, radius=50.0, sigma=20.0):
    if not gps_nodes or not segments:
        raise ValueError("Нужны GPS-точки и дорожные сегменты")
    layers = get_candidates(gps_nodes, segments, radius, sigma)
    graph = build_road_graph(segments, layers)

    @lru_cache(maxsize=64)
    def routes_from(vertex):
        return dijkstra(graph, vertex)

    matches, route_edges, breaks = [None] * len(gps_nodes), [], []
    indices, parents, scores = [], [], []

    def finish_block():
        if not indices:
            return
        best = max(range(len(scores)), key=lambda j: scores[j])
        for position in range(len(indices) - 1, -1, -1):
            matches[indices[position]] = layers[indices[position]][best]
            best = parents[position][best]
        #восстанавливаем дорожный путь между выбранными кандидатами
        for i, j in zip(indices, indices[1:]):
            source, target = matches[i]["vertex"], matches[j]["vertex"]
            _, previous = routes_from(source)
            path = []
            #идем от конца пути к началу
            while target != source:
                target, edge = previous[target]
                path.append(edge)
            route_edges.extend(reversed(path))

    #обрабатываем слои кандидатов
    for i, layer in enumerate(layers):
        if not layer:
            continue
        if indices:
            old_i = indices[-1]
            new_scores, new_parents = [-math.inf] * len(layer), [None] * len(layer)
            for k, a in enumerate(layers[old_i]):
                if not math.isfinite(scores[k]):
                    continue
                distances, _ = routes_from(a["vertex"])
                #пробуем переход к каждому кандидату текущего слоя
                for j, b in enumerate(layer):
                    transition = transition_score(gps_nodes[old_i], gps_nodes[i], a, b,
                                                  distances.get(b["vertex"], math.inf))
                    total = scores[k] + transition + b["log_observation"]
                    #запоминаем лучший способ прийти к кандидату b
                    if total > new_scores[j]:
                        new_scores[j], new_parents[j] = total, k
            #если хотя бы один переход возможен то продолжаем текущий блок
            if any(math.isfinite(score) for score in new_scores):
                indices.append(i)
                parents.append(new_parents)
                scores = new_scores
                continue
            finish_block()
            breaks.append((old_i, i))
        indices = [i]
        parents = [[None] * len(layer)]
        scores = [candidate["log_observation"] for candidate in layer]
    finish_block()
    return matches, route_edges, breaks


def draw_map(points_xy, segments, gps, matches, route_edges):
    fig, ax = plt.subplots(figsize=(11, 9))
    lines = [[point_on_segment(s, 0), point_on_segment(s, 1)] for s in segments.values()]
    ax.add_collection(LineCollection(lines, colors="0.75", linewidths=1, label="Дороги"))
    ax.scatter(*zip(*points_xy.values()), color="0.4", s=5, label="Точки OSM")
    ax.plot([p["x"] for p in gps], [p["y"] for p in gps], "o--", color="crimson",
            markersize=4, zorder=4, label="Исходная GPS-траектория")
    lines = [[point_on_segment(segments[e["segment_id"]], e["t0"]),
              point_on_segment(segments[e["segment_id"]], e["t1"])] for e in route_edges]
    if lines:
        ax.add_collection(LineCollection(lines, colors="forestgreen", linewidths=3,
                                        label="Привязанный путь по дорогам"))
    selected = [p for p in matches if p is not None]
    if selected:
        ax.scatter([p["x"] for p in selected], [p["y"] for p in selected],
                   color="royalblue", s=25, zorder=5, label="Привязанные точки")
    visible = [(p["x"], p["y"]) for p in gps + selected]
    for line in lines:
        visible.extend(line)
    xs, ys = zip(*visible)
    margin = max(40, 0.15 * max(max(xs) - min(xs), max(ys) - min(ys)))
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    title = "Привязка фиксированного учебного списка GPS-точек"
    ax.set(xlabel="X, м (UTM 37N)", ylabel="Y, м (UTM 37N)", title=title)
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not GPS_LAT_LON:
        raise ValueError("!!Заполни GPS_LAT_LON!!")
    points_xy, segments = parse_osm(MAP_FILE)
    gps_nodes = []
    for lat, lon in GPS_LAT_LON:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Неверный формат GPS")
        x, y = TRANSFORMER.transform(lon, lat, errcheck=True)
        gps_nodes.append({"x": x, "y": y})
    matched_points, route_edges, breaks = match_gps(
        gps_nodes, segments, SEARCH_RADIUS_M, GPS_SIGMA_M)
    draw_map(points_xy, segments, gps_nodes, matched_points, route_edges)
