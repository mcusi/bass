from util.context import context


class EventMeta():
    """Keeps track of neural network proposal used to initialize Event"""
    def __init__(self, hypothesis):
        self.proposal_rank = hypothesis["meta"]["rank"]
        self.ious = hypothesis["meta"]["ious"]
        self.seg_score = hypothesis["meta"]["score"]
        self.round_idx = hypothesis["meta"].get("round_idx", None)
        self.group_onset = hypothesis["meta"].get("group_onset", None)
        # Sounds won't have a concurrent rank if there wasn't any max_concurrent trimming
        self.concurrent_rank = hypothesis["meta"].get("concurrent_rank", None)
        self.background_proposal = hypothesis["meta"].get("background_proposal", False)  # only background proposal has it

    def compare(self, other_meta):
        if other_meta.get("background_proposal", False) or self.background_proposal:
            return True
        else:
            iou = other_meta["ious"][self.proposal_rank]
            return iou < context.hypothesis["neural"]["iou_threshold"]


class SceneMeta():
    """Keeps track of sequential inference proposals"""
    def __init__(self):
        self.history = []
        self.next_round()

    def next_round(self, round_type=None):
        self.history.append({'sources': [], 'events': [], 'round_type':round_type})

    def add_source(self, source):
        self.history[-1]['sources'].append(source)

    def add_event(self, event):
        self.history[-1]['events'].append(event)

    @property
    def round_type(self):
        return self.history[-1]['round_type']

    @property
    def new_sources(self):
        return self.history[-1]['sources']

    @property
    def new_events(self):
        return self.history[-1]['events']

    @property
    def age(self):
        return len(self.history) - 1
