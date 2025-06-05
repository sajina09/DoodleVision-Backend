import random

# Pre-written story options for each class, because I couldn't get the LLM model to work on free hosting sites
STORY_POOL = {
    "flower": [
        "Once a little flower danced in the wind, smiling at the sun.",
        "A flower bloomed in the middle of a snowy day, surprising everyone.",
        "The flower whispered secrets to the bees who visited her."
    ],
    "hat": [
        "The magical hat flew away on an adventure without its owner.",
        "A tiny mouse made a cozy home inside an old top hat.",
        "The hat gave anyone who wore it a silly laugh."
    ],
    "bicycle": [
        "The bicycle dreamed of riding across the rainbow.",
        "One day, the bike raced the wind and almost won.",
        "The bicycle took a nap under a tree after a long ride."
    ],
    "cat": [
        "The cat chased shadows all morning and finally caught one.",
        "A sleepy cat guarded her favorite sunbeam like treasure.",
        "This cat loved wearing socks and doing flips."
    ],
    "tree": [
        "A tree grew so tall it tickled the clouds.",
        "Birds sang lullabies to the tree every night.",
        "The tree told bedtime stories to the squirrels."
    ],
    "fish": [
        "A fish made bubbles that looked like stars.",
        "The fish swam in circles just for fun.",
        "One fish dreamed of flying like a bird."
    ],
    "candle": [
        "The candle lit up a little snail’s birthday party.",
        "The candle glowed bright even during a storm.",
        "This candle smelled like cookies and giggles."
    ],
    "star": [
        "A star blinked three times and granted a tiny wish.",
        "This star liked to twirl and play hide and seek.",
        "One night, a star fell into a child’s dream."
    ],
    "face": [
        "The happy face winked every time someone smiled.",
        "A silly face popped up in the mirror and giggled.",
        "The face changed colors based on your mood."
    ],
    "house": [
        "The little house danced every time it rained.",
        "This house had a door that loved to sing.",
        "At night, the house told jokes to the moon."
    ]
}

def get_random_story(label: str) -> str:
    label = label.lower()
    return random.choice(STORY_POOL.get(label, ["This is a mysterious object with no tale."]))

