"""中文 TinyStories 风格语料生成器。
受限词汇 + 多模板组合，造出"有角色一致性、情节完整"的短故事。
关键：故事内部主角名贯穿全文 -> 用于检验模型是否真学会"角色一致"(压缩率会奖励一致性)。
不依赖外部 API，本地可造，可复现(seed)。
"""
import random

# 受限词汇库（3-4岁能懂级别，对应 TinyStories 思想）
NAMES = ["小明", "莉莉", "丁丁", "小兔", "小熊", "小猫", "妞妞", "乐乐", "圆圆", "贝贝",
         "琪琪", "亮亮", "果果", "葵葵", "岱岱", "杉杉", "涛涛", "兰兰"]
PLACES = ["公园", "森林", "河边", "草地", "家里", "山上", "花园", "湖边",
          "海边", "学校", "菜园", "树下"]
OBJECTS = ["皮球", "蝴蝶", "苹果", "风筝", "小花", "石头", "气球", "贝壳",
           "果子", "红果", "金鱼", "小草"]
FEELINGS = ["很开心", "有点难过", "很惊喜", "很自豪", "很温暖", "笑了", "很满足", "很高兴"]
HELPERS = ["妈妈", "朋友", "老师", "爷爷", "小狗", "哥哥", "姐姐", "奶奶"]


def _story(rng, names):
    """一个完整模板故事：主角(贯穿) + 地点 + 想要的东西 + 障碍 + 帮助者 + 解决 + 情感收尾。
    names 参数化：训练集与 unseen-G 用不同的名字集(结构化留出)。"""
    name = rng.choice(names)        # 主角，全文唯一，贯穿
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    helper = rng.choice(HELPERS)
    feel = rng.choice(FEELINGS)
    templates = [
        f"从前有一个叫{name}的孩子。有一天，{name}来到{place}，看见了一个{obj}。"
        f"{name}很想要那个{obj}，可是够不到。这时候，{helper}来了，帮{name}一起拿到了{obj}。"
        f"{name}{feel}，对{helper}说谢谢。最后{name}高高兴兴地回家了。",

        f"{name}是一个可爱的孩子。这天，{name}在{place}玩。{name}发现了一个{obj}，"
        f"想把它带回家。但是{obj}太重了，{name}搬不动。{helper}看见了，就来帮忙。"
        f"他们一起把{obj}搬回了家。{name}{feel}。",

        f"早上，{name}和{helper}一起去{place}。{name}想找一个{obj}。"
        f"找了很久都没找到，{name}有点着急。后来{helper}说：别急，我们一起找。"
        f"终于，{name}在草丛里找到了{obj}。{name}{feel}，笑了。",
    ]
    return rng.choice(templates)


def make_corpus(n_stories=8000, seed=0, names=None):
    """names=None 时用全部 NAMES。结构化留出时传入不同名字子集。"""
    rng = random.Random(seed)
    pool = names if names is not None else NAMES
    stories = [_story(rng, pool) for _ in range(n_stories)]
    return "\n".join(stories)


# 结构化留出：训练集与留出集用不同主角名（防死记：记不住没见过的名字，只能学会“主角槽位”规律）
TRAIN_NAMES = NAMES[:12]
HOLDOUT_NAMES = NAMES[12:]


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    import torch
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


if __name__ == "__main__":
    # 自检：seen 与 unseen-G 用不同 seed 造，词表一致，但具体故事组合不同
    seen = make_corpus(2000, seed=0)
    unseen = make_corpus(500, seed=999)
    print("seen chars:", len(seen), "unseen chars:", len(unseen))
    print("sample:\n", seen[:120])
    stoi, _ = build_vocab(seen + unseen)
    print("vocab size:", len(stoi))
