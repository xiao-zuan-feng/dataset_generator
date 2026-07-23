import json
import random
import os


class GSM8KPicker:
    def __init__(self, jsonl_file, record_file="picked_ids.txt", no_repeat=False):
        self.jsonl_file = jsonl_file
        self.record_file = record_file
        self.total_lines = self._count_lines()
        self.picked_ids = self._load_picked_ids()
        self.no_repeat = no_repeat

    def _count_lines(self):
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _load_picked_ids(self):
        if os.path.exists(self.record_file):
            with open(self.record_file, 'r', encoding='utf-8') as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()

    def _save_picked_ids(self):
        with open(self.record_file, 'w', encoding='utf-8') as f:
            for pid in sorted(self.picked_ids):
                f.write(f"{pid}\n")

    def get_unpicked_ids(self):
        all_ids = set(range(self.total_lines))
        return list(all_ids - self.picked_ids)

    def pick_one(self):
        if self.no_repeat:
            available_ids = self.get_unpicked_ids()
            if not available_ids:
                return None
            selected_id = random.choice(available_ids)
            self.picked_ids.add(selected_id)
            self._save_picked_ids()
        else:
            selected_id = random.randint(0, self.total_lines - 1)

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == selected_id:
                    data = json.loads(line)
                    break

        return data['question']


class ShareGPTPicker:
    def __init__(self, jsonl_file, no_repeat=False):
        self.jsonl_file = jsonl_file
        self.no_repeat = no_repeat
        self.all_data = self._load_all()
        self.picked_indices = []

    def _load_all(self):
        data_list = []
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    data_list.append(obj)
                except json.JSONDecodeError:
                    continue
        return data_list

    def pick_one(self):
        total = len(self.all_data)
        if total == 0:
            return None

        if self.no_repeat:
            available = [i for i in range(total) if i not in self.picked_indices]
            if not available:
                return None
            idx = random.choice(available)
            self.picked_indices.append(idx)
        else:
            idx = random.randint(0, total - 1)

        return self.all_data[idx]
