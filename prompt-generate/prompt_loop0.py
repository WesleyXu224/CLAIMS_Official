import argparse
import csv
import os
import random
import re

def generate_random_sports_prompt(num_prompts):
    """
    随机生成指定数量的不重复运动prompt（移除场景和方向变量，优化描述格式）
    参数:
        num_prompts: 生成的prompt数量（需确保不超过最大可能组合数）
    返回:
        不重复的运动prompt列表
    """
    # 1. 优化后的运动无级变量库（移除数值和括号，删除scene和direction）
    sports_vars = {
        "base_action": [
            # Simple (Low-Dynamic)
            "jogging with steady pace and relaxed arm swing",
            "basic jump with two-foot takeoff and landing",
            "push-up with knee support and slow movement",
            "soccer forward pass with slow arm swing",
            "walking lunges with steady step and bent knees",
            "static plank with forearm support and straight body",
            "basketball chest pass with close range and slow release",
            
            # Medium-Difficulty
            "sprint start with three-point stance and moderate acceleration",
            "vertical jump with countermovement and controlled landing",
            "lunge with dynamic step and proper knee alignment",
            "basketball dribble with steady bounce and low height",
            "tennis forehand with moderate swing and controlled range",
            "swimming freestyle with steady pace and consistent stroke",
            "cycling with moderate speed and regular pedal rotation",
            
            # High-Dynamic/High-Difficulty
            "100m sprint with maximum acceleration and rapid steps",
            "long jump with fast approach run and airborne flight",
            "high jump with Fosbury flop technique and back arch",
            "soccer kick with full instep strike and long range",
            "burpee with rapid transitions between movements",
            "400m hurdles with consistent rhythm between obstacles",
            "basketball slam dunk with high vertical jump and one-hand grip",
            "parkour vault with fast runup and smooth obstacle clearance",
            "snowboard half-pipe with high airborne and spinning movement",
            "rugby tackle with explosive drive and low center of gravity",
            "tennis serve with high speed and topspin rotation",
            "triathlon transition with seamless switch between disciplines",
            "volleyball spike with high jump and powerful downward strike",
            "skateboard ollie with board flip and balanced landing",
            "hurdle race with rapid steps between obstacles",
            "martial arts kickboxing combo with quick successive strikes"
        ],
        "combo_action": [
            # Simple Combos
            "jogging → basic jump → walking recovery",
            "push-up → walking lunges → static plank",
            "basketball chest pass → slow dribble → chest pass",
            
            # Medium-Difficulty Combos
            "sprint start → short acceleration → slow stop",
            "tennis forehand → backhand → slow run to net",
            "swimming freestyle → turn → freestyle continuation",
            
            # High-Dynamic Combos
            "sprint start → rapid acceleration → slide stop",
            "dribble → crossover → jump shot → rebound",
            "long jump approach → takeoff → landing → recovery",
            "400m hurdles → sprint finish → cool-down jog",
            "parkour vault → run → ollie → landing roll",
            "volleyball set → jump spike → defensive block",
            "rugby tackle → standup → pass → sprint"
        ],
        "detail": [
            # Basic Details
            "jogging with consistent stride length and parallel arm swing",
            "push-up with elbows at proper angle and tight core",
            "forward pass with hands positioned on ball sides and follow-through",
            "plank with elbows under shoulders and engaged glutes",
            
            # Medium-Difficulty Details
            "sprint start with front knee over toe and bent back leg",
            "dribble with spread fingers and ball contact with pads",
            "tennis forehand with proper racket angle and fluid swing",
            
            # High-Difficulty Details
            "long jump takeoff with extended front leg and tucked back leg",
            "Fosbury flop with arched back and hips positioned over bar",
            "soccer kick with properly placed plant foot and hip rotation",
            "slam dunk with powerful approach and high hand placement",
            "parkour speed vault with balanced hand support and parallel leg swing",
            "snowboard half-pipe with weight shifted forward and shoulder-initiated spin",
            "tennis serve with high toss and optimal impact position",
            "volleyball spike with two-leg jump and cocked arm position"
        ],
        "speed_rhythm": [
            # Simple Rhythm
            "jogging with consistent pace and regular step frequency",
            "push-up with slow descent and controlled ascent",
            "walking lunges with steady tempo and no speed variation",
            
            # Medium-Difficulty Rhythm
            "sprint with gradual acceleration and maintained speed",
            "dribble with steady bounce rate followed by quick pass",
            "swimming with consistent stroke timing and efficient turns",
            
            # High-Dynamic Rhythm
            "100m sprint with rapid acceleration and sustained maximum speed",
            "burpee with quick transitions between push-up and jump",
            "parkour with fast runup and rapid sequence execution",
            "tennis serve with quick toss and explosive swing",
            "hurdle race with rapid sprint between obstacle jumps"
        ]
    }

    # 2. 优化后的运动无级模板库（移除scene和direction相关内容）
    sports_templates = [
        "The athlete performed {combo_action} with {detail}, maintaining {speed_rhythm}.",
        "During training, the sportsperson executed {base_action} focusing on {detail}, following {speed_rhythm}.",
        "The athlete completed {combo_action} with {detail} (power control), adhering to {speed_rhythm}.",
        "In the key moment, the athlete transitioned from {combo_action} to {base_action} with {detail}, using {speed_rhythm}.",
        "The sportsperson combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The sequence began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the athlete practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In the final phase, the athlete executed {base_action} followed by {combo_action} with {detail}, optimized for {speed_rhythm}."
    ]

    # 3. 计算最大可能的不重复组合数
    max_possible = 0
    for template in sports_templates:
        has_combo = "{combo_action}" in template
        has_base = "{base_action}" in template
        
        if has_combo and has_base:
            # 同时包含组合动作和基础动作
            max_possible += len(sports_vars["combo_action"]) * len(sports_vars["base_action"]) * len(sports_vars["detail"]) * len(sports_vars["speed_rhythm"])
        elif has_combo:
            # 仅包含组合动作
            max_possible += len(sports_vars["combo_action"]) * len(sports_vars["detail"]) * len(sports_vars["speed_rhythm"])
        elif has_base:
            # 仅包含基础动作
            max_possible += len(sports_vars["base_action"]) * len(sports_vars["detail"]) * len(sports_vars["speed_rhythm"])
    
    # 校验输入数量合理性
    if num_prompts > max_possible:
        raise ValueError(f"生成数量超过最大可能组合数（最大{max_possible}个）")
    if num_prompts <= 0:
        raise ValueError("生成数量必须为正整数")

    # 4. 随机生成不重复的prompt
    generated = set()
    while len(generated) < num_prompts:
        template = random.choice(sports_templates)
        
        vars_selected = {
            "detail": random.choice(sports_vars["detail"]),
            "speed_rhythm": random.choice(sports_vars["speed_rhythm"])
        }
        
        if "{combo_action}" in template:
            vars_selected["combo_action"] = random.choice(sports_vars["combo_action"])
        if "{base_action}" in template:
            vars_selected["base_action"] = random.choice(sports_vars["base_action"])
        
        prompt = template
        for key, value in vars_selected.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        generated.add(prompt)
    
    return list(generated)

def generate_random_gymnastics_prompt(num_prompts):
    """
    随机生成指定数量的不重复体操prompt（移除场景和方向，优化变量描述格式）
    参数:
        num_prompts: 生成的prompt数量（需确保不超过最大可能组合数）
    返回:
        不重复的体操prompt列表
    """
    # 1. 优化后的体操无级变量库（移除数值/括号，删除scene和direction）
    gymnastics_vars = {
        "base_action": [
            # Simple (Low-Dynamic)
            "basic squat with bent knees and flat feet",
            "simple arm stretch with slow overhead reach",
            "balance beam walk with steady pace and outstretched arms",
            "vault approach with straight walk",
            "static plank with forearm support and straight body",
            "slow leg lift with sideways movement and controlled lowering",
            "parallel bars hang with straight arms",
            
            # Medium-Difficulty
            "cartwheel with sideways movement and two-hand support",
            "single leg lift with static balance on beam",
            "floor jump with vertical leap and soft landing",
            "rings hang with slow shoulder shrugs",
            "saddle horse mount with slow leg swing and steady grip",
            "uneven bars pull-up with controlled ascent",
            "floor roll with forward movement",
            
            # High-Dynamic/High-Difficulty
            "double backflip with airborne rotation and tuck position",
            "balance beam split leap with leg extension and airborne movement",
            "vault tsukahara with springboard takeoff and twisting movement",
            "parallel bars swing with giant circle and momentum-based movement",
            "triple backflip with consecutive airborne spins and pike position",
            "balance beam back handspring with no-hand flip and rotation",
            "vault yurchenko with round-off onto springboard and twisting movement",
            "rings iron cross with horizontal arm hold and core tension",
            "parallel bars dismount with double backflip off bars",
            "floor arabesque leap with extended leg and airborne movement",
            "uneven bars kip with dynamic swing to support",
            "saddle horse double leg circle with rapid leg swing",
            "floor full-twisting double backflip with spins and twisting movement",
            "balance beam front handspring into split leap with seamless transition",
            "rings swing to cross with dynamic swing into iron cross"
        ],
        "combo_action": [
            # Simple Combos
            "basic squat → simple arm stretch → floor jump",
            "balance beam walk → single leg lift → step down",
            "plank → slow leg lift → arm stretch",
            
            # Medium-Difficulty Combos
            "cartwheel → floor roll → standing jump",
            "uneven bars pull-up → hang → slow swing",
            "saddle horse mount → single leg circle → dismount",
            
            # High-Dynamic Combos
            "vault approach → tsukahara → landing roll",
            "floor cartwheel → double backflip → split leap",
            "triple backflip → floor arabesque leap → full-twisting backflip",
            "uneven bars kip → swing → parallel bars dismount",
            "rings swing → iron cross → dynamic dismount",
            "balance beam back handspring → split leap → front handspring"
        ],
        "detail": [
            # Basic Details
            "squat with knees over toes, straight back and forward-facing arms",
            "beam walk with steady steps, forward gaze and relaxed shoulders",
            "plank with elbows under shoulders, engaged core and no hip sag",
            "hang with straight arms, shoulders away from ears and firm grip",
            
            # Medium-Difficulty Details
            "cartwheel with shoulder-width hand placement, straight legs and together landing feet",
            "pull-up with chin over bar and fully extended elbows at bottom",
            "floor roll with tucked chin, shoulder-initiated rotation and no head impact",
            
            # High-Difficulty Details
            "double backflip with knees tucked to chest at peak height and timely untucking before landing",
            "split leap with squared hips, pointed toes and straight back",
            "tsukahara with arched back at takeoff and twisting movement initiated mid-air",
            "triple backflip with strong leg drive at takeoff and maintained pike position",
            "iron cross with arms parallel to ground, proper shoulder alignment and locked core",
            "yurchenko vault with hands on springboard during round-off and post-takeoff twist",
            "full-twisting backflip with twist initiation after first spin and fixed gaze on landing",
            "beam back handspring with heel-driven takeoff, arched body mid-air and ball-of-foot landing"
        ],
        "speed_rhythm": [
            # Simple Rhythm
            "balance beam walk with slow steps and steady arm movement",
            "squat with controlled descent, held position and steady ascent",
            "plank with held position, rest period and repetition",
            
            # Medium-Difficulty Rhythm
            "cartwheel with timely completion, rest period and floor roll",
            "pull-up with controlled ascent, held position and steady descent",
            "balance beam walk with steady steps and held leg lift",
            
            # High-Dynamic Rhythm
            "vault approach with progressive speed increase and explosive takeoff",
            "swing with gradual buildup, sudden release for dismount and increased speed",
            "backflip sequence with quick takeoff, airborne spins and fast landing",
            "rings swing with steady forward and backward movement and explosive transition to cross",
            "balance beam combo with quick back handspring, split leap and front handspring"
        ]
    }

    # 2. 优化后的体操无级模板库（移除scene和direction相关内容）
    gymnastics_templates = [
        "The gymnast performed {combo_action} with {detail}, following {speed_rhythm}.",
        "During practice, the athlete executed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The gymnast completed {combo_action} with {detail} (airborne control), adhering to {speed_rhythm}.",
        "In the dismount sequence, the athlete executed {base_action} followed by {combo_action} with {detail}, using {speed_rhythm}.",
        "The routine began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the gymnast practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In competition, the gymnast combined {base_action} and {combo_action} with {detail}, optimized for {speed_rhythm}.",
        "The athlete transitioned from {combo_action} to {base_action} with {detail}, synchronizing with {speed_rhythm}."
    ]

    # 3. 计算最大可能的不重复组合数
    max_possible = 0
    for template in gymnastics_templates:
        has_combo = "{combo_action}" in template
        has_base = "{base_action}" in template
        
        if has_combo and has_base:
            # 同时包含组合动作和基础动作
            max_possible += len(gymnastics_vars["combo_action"]) * len(gymnastics_vars["base_action"]) * len(gymnastics_vars["detail"]) * len(gymnastics_vars["speed_rhythm"])
        elif has_combo:
            # 仅包含组合动作
            max_possible += len(gymnastics_vars["combo_action"]) * len(gymnastics_vars["detail"]) * len(gymnastics_vars["speed_rhythm"])
        elif has_base:
            # 仅包含基础动作
            max_possible += len(gymnastics_vars["base_action"]) * len(gymnastics_vars["detail"]) * len(gymnastics_vars["speed_rhythm"])
    
    # 校验输入数量合理性
    if num_prompts > max_possible:
        raise ValueError(f"生成数量超过最大可能组合数（最大{max_possible}个）")
    if num_prompts <= 0:
        raise ValueError("生成数量必须为正整数")

    # 4. 随机生成不重复的prompt
    generated = set()
    while len(generated) < num_prompts:
        template = random.choice(gymnastics_templates)
        
        vars_selected = {
            "detail": random.choice(gymnastics_vars["detail"]),
            "speed_rhythm": random.choice(gymnastics_vars["speed_rhythm"])
        }
        
        if "{combo_action}" in template:
            vars_selected["combo_action"] = random.choice(gymnastics_vars["combo_action"])
        if "{base_action}" in template:
            vars_selected["base_action"] = random.choice(gymnastics_vars["base_action"])
        
        prompt = template
        for key, value in vars_selected.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        generated.add(prompt)
    
    return list(generated)

def generate_random_martial_arts_prompt(num_prompts):
    """
    随机生成指定数量的不重复武术prompt（移除场景和方向，优化变量描述格式）
    参数:
        num_prompts: 生成的prompt数量（需确保不超过最大可能组合数）
    返回:
        不重复的武术prompt列表
    """
    # 1. 优化后的武术无级变量库（移除数值/括号，删除scene和direction）
    martial_arts_vars = {
        "base_action": [
            # Simple (Low-Dynamic)
            "basic bow stance with static posture and shoulder-width feet",
            "simple cross fist with slow arm swing and minimal hip movement",
            "basic horse stance with steady posture and bent knees",
            "simple palm strike with straight arm and slow retraction",
            "static mountain stance with parallel feet and even weight distribution",
            "slow downward chop with arm swing from shoulder and low speed",
            "basic front kick with knee-height movement and slow extension",
            
            # Medium-Difficulty
            "standard hook punch with moderate hip rotation and steady strike",
            "front kick with waist-height movement and controlled landing",
            "side stance shift with smooth weight transfer",
            "basic push hands with gentle force redirection and slow reaction",
            "low sweep kick with ankle-height movement and slow leg swing",
            "single whip stance with dynamic weight shift and steady transition",
            "basic elbow strike with short range and moderate force",
            
            # High-Dynamic/High-Difficulty
            "whirlwind kick with spinning movement, airborne leg extension and rapid rotation",
            "double kick with rapid consecutive leg strikes and no pause",
            "iron bridge with backbend posture, arm support and core tension",
            "snake-like fist with rapid wrist flick and zigzag arm movement",
            "triple spinning kick with consecutive spins and airborne leg extension",
            "flying side kick with airborne lateral movement and focused strike",
            "reverse tornado kick with backward spinning movement and heel strike",
            "leaping tiger claw with forward jump, extended fingers and grappling focus",
            "dragon tail whip with rapid leg swing from low to high and arcing movement",
            "jumping double palm strike with airborne posture and extended body",
            "side flip kick with lateral flip movement, leg strike and no hand support",
            "shadowless kick with rapid low movement and barely visible leg",
            "eight-directional step punch with dynamic stepping and multi-angle strikes",
            "ground sweep to standup with seamless transition from prone to jumping kick"
        ],
        "combo_action": [
            # Simple Combos
            "basic bow stance → simple cross fist → horse stance",
            "front kick (knee-height) → basic palm strike → side stance shift",
            "mountain stance → downward chop → static hold",
            
            # Medium-Difficulty Combos
            "hook punch → low sweep kick → push hands",
            "side stance shift → elbow strike → single whip stance",
            "front kick (waist-height) → cross fist → backward step",
            
            # High-Dynamic Combos
            "whirlwind kick → landing hook punch → double kick",
            "push hands → snake-like fist → iron bridge transition",
            "triple spinning kick → flying side kick → reverse tornado kick",
            "leaping tiger claw → dragon tail whip → jumping double palm strike",
            "ground sweep → standup → shadowless kick → eight-directional step punch"
        ],
        "detail": [
            # Basic Details
            "bow stance with bent knees, straight back and forward-facing toes",
            "cross fist with fists crossed at chest and outward-facing knuckles",
            "horse stance with wide foot placement, engaged core and no sway",
            "palm strike with together fingers, flat palm and impact at palm heel",
            
            # Medium-Difficulty Details
            "hook punch with bent elbow, force from torso twist and target-focused landing",
            "push hands with relaxed arms, opponent force following and no rigid resistance",
            "sweep kick with straight leg, pointed foot and contact at opponent's lower leg",
            
            # High-Difficulty Details
            "whirlwind kick with pivot foot rotation, tight core for balance and target-focused gaze",
            "double kick with first strike to thigh, second to waist and no pause between",
            "iron bridge with locked elbows, squeezed shoulder blades and steady breath",
            "triple spinning kick with back foot push for spin initiation and extended legs mid-air",
            "flying side kick with back leg takeoff, straight front leg and landing on takeoff leg",
            "reverse tornado kick with arm swing for spin initiation and heel strike to opponent's torso",
            "shadowless kick with low leg position, instep strike and rapid retraction"
        ],
        "speed_rhythm": [
            # Simple Rhythm
            "stance transitions with slow movement and no sudden speed change",
            "punching with steady speed and consistent force",
            "stance hold → slow strike → reset with steady pace",
            
            # Medium-Difficulty Rhythm
            "alternating speed: slow punch → moderate kick → slow step",
            "reaction with moderate speed in push hands practice",
            "preparation → steady strike → recovery with balanced timing",
            
            # High-Dynamic Rhythm
            "stance buildup with slow movement → explosive kick → rapid recovery",
            "push hands with slow movement → sudden snake-like fist with burst speed",
            "spinning with accelerating speed: initial slow spin → gradual speed increase",
            "jump-strike with slow approach → explosive takeoff → airborne strike → quick landing",
            "ground-to-air transition with slow sweep → rapid standup → explosive kick"
        ]
    }

    # 2. 优化后的武术无级模板库（移除scene和direction相关内容）
    martial_arts_templates = [
        "The martial artist executed {combo_action} with {detail}, following {speed_rhythm}.",
        "During training, the practitioner performed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The martial artist completed {combo_action} with {detail} (power control), adhering to {speed_rhythm}.",
        "In the combat sequence, the martial artist transitioned from {combo_action} to {base_action} with {detail}, using {speed_rhythm}.",
        "The practitioner combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The routine began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During form practice, the practitioner performed {base_action} with {detail}, following {speed_rhythm}.",
        "In dynamic practice, the martial artist executed {base_action} followed by {combo_action} with {detail}, optimized for {speed_rhythm}."
    ]

    # 3. 计算最大可能的不重复组合数
    max_possible = 0
    for template in martial_arts_templates:
        has_combo = "{combo_action}" in template
        has_base = "{base_action}" in template
        
        if has_combo and has_base:
            # 同时包含组合动作和基础动作
            max_possible += len(martial_arts_vars["combo_action"]) * len(martial_arts_vars["base_action"]) * len(martial_arts_vars["detail"]) * len(martial_arts_vars["speed_rhythm"])
        elif has_combo:
            # 仅包含组合动作
            max_possible += len(martial_arts_vars["combo_action"]) * len(martial_arts_vars["detail"]) * len(martial_arts_vars["speed_rhythm"])
        elif has_base:
            # 仅包含基础动作
            max_possible += len(martial_arts_vars["base_action"]) * len(martial_arts_vars["detail"]) * len(martial_arts_vars["speed_rhythm"])
    
    # 校验输入数量合理性
    if num_prompts > max_possible:
        raise ValueError(f"生成数量超过最大可能组合数（最大{max_possible}个）")
    if num_prompts <= 0:
        raise ValueError("生成数量必须为正整数")

    # 4. 随机生成不重复的prompt
    generated = set()
    while len(generated) < num_prompts:
        template = random.choice(martial_arts_templates)
        
        vars_selected = {
            "detail": random.choice(martial_arts_vars["detail"]),
            "speed_rhythm": random.choice(martial_arts_vars["speed_rhythm"])
        }
        
        if "{combo_action}" in template:
            vars_selected["combo_action"] = random.choice(martial_arts_vars["combo_action"])
        if "{base_action}" in template:
            vars_selected["base_action"] = random.choice(martial_arts_vars["base_action"])
        
        prompt = template
        for key, value in vars_selected.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        generated.add(prompt)
    
    return list(generated)
def generate_random_dance_prompt(num_prompts):
    """
    随机生成指定数量的不重复舞蹈prompt（移除场景和方向，优化变量描述格式）
    参数:
        num_prompts: 生成的prompt数量（需确保不超过最大可能组合数）
    返回:
        不重复的舞蹈prompt列表
    """
    # 1. 优化后的舞蹈无级变量库（移除数值/括号，删除scene和direction）
    dance_vars = {
        "base_action": [
            # Simple (Low-Dynamic)
            "basic tendu with slow leg slide and small movement",
            "simple plié with gentle knee bend and steady rhythm",
            "basic port de bras with slow arm sweep and no torso twist",
            "simple step with flat foot and minimal weight shift",
            "static arabesque with held posture and minimal sway",
            "slow arm circle with forward rotation and steady movement",
            "heel-toe tap with slow alternating motion and no weight transfer",
            
            # Medium-Difficulty
            "relevé with slow rise onto toes and controlled balance",
            "chassé with gliding step and moderate speed",
            "pirouette with single turn and steady rotation",
            "grand battement with medium leg lift and controlled descent",
            "sauté with small jump and two-foot takeoff landing",
            "pas de chat with cat-like step and moderate height",
            "chainé turns with continuous slow spins and steady rotation",
            "lunge with arm extension and balanced posture",
            
            # High-Dynamic/High-Difficulty
            "fouetté turn with rapid spinning, leg flick and high balance demand",
            "grand jeté with leap, split posture and airborne extension",
            "pas de bourrée with quick footwork sequence and fast weight shifts",
            "contraction-release with abrupt torso twist and dynamic core control",
            "triple pirouette with consecutive spins and spotting technique",
            "aerial cartwheel with no-hand airborne flip and rotation",
            "grand allegro with high leap, leg extension and airborne movement",
            "bourrée en pointe with rapid tiptoe steps and light footwork",
            "saut de basque with turning leap, leg swing and full rotation",
            "port de bras with backbend, dynamic spinal arch and synchronized arm sweep",
            "jeté battu with rapid alternating leg beats mid-air",
            "tour en l'air with airborne full rotation and tucked position",
            "pique turn chain with continuous sharp turns on pointe",
            "floorwork roll into jump with seamless transition from prone to aerial"
        ],
        "combo_action": [
            # Simple Combos
            "basic tendu → simple plié → basic port de bras",
            "simple step → relevé → slow arm drop",
            "static arabesque → heel-toe tap → arm circle",
            "sauté → slow lunge → port de bras",
            
            # Medium-Difficulty Combos
            "chassé → single pirouette → grand battement",
            "chainé turns → pas de chat → relevé hold",
            "lunge with arm extension → sauté → side step",
            
            # High-Dynamic Combos
            "pas de bourrée → fouetté turn → grand jeté",
            "contraction-release → grand battement → chassé → pirouette",
            "triple pirouette → aerial cartwheel → grand allegro",
            "saut de basque → jeté battu → bourrée en pointe",
            "floorwork roll → tour en l'air → pique turn chain → landing plié"
        ],
        "detail": [
            # Basic Details
            "tendu with pointed toes and lifted heel",
            "arm movement with relaxed shoulders and slightly bent elbows",
            "plié with appropriate depth and evenly distributed weight",
            "arabesque with straight back leg, aligned hips and squared shoulders",
            "step with heel strike first and smooth toe follow-through",
            
            # Medium-Difficulty Details
            "pirouette with fixed point spotting, engaged core and no sway",
            "chassé with feet barely leaving ground and smooth gliding motion",
            "sauté with bent knees on takeoff and soft absorption on landing",
            "chainé turns with arms in first position and rotation from torso",
            
            # High-Difficulty Details
            "fouetté turn with head spotting for direction, engaged core and controlled leg flick",
            "grand jeté with airborne split, fully extended legs and straight back",
            "pas de bourrée with rapid foot taps and minimal upper body movement",
            "triple pirouette with back foot push, maintained turnout and steady rotation",
            "aerial cartwheel with shoulder rotation for takeoff and elevated hips mid-air",
            "grand allegro with extended airborne time, fully stretched legs and balanced posture",
            "tour en l'air with timely tuck initiation and smooth untuck before landing",
            "pique turn chain with pointe foot aligned to hip and arm swing for turn initiation"
        ],
        "speed_rhythm": [
            # Simple Rhythm
            "steady tempo with consistent movement speed",
            "slow leg movement matched to arm speed",
            "held posture → slow movement → held posture with balanced timing",
            "half-time steps with deliberate transitions",
            
            # Medium-Difficulty Rhythm
            "alternating count movements with varied timing",
            "moderate tempo with brief pauses",
            "slow step → medium turn → slow step with smooth transitions",
            
            # High-Dynamic Rhythm
            "slow contraction → sudden release with speed burst",
            "slow steps → fast pirouette → suspended landing with dynamic contrast",
            "accelerating turns with gradual speed increase",
            "leap sequence with slow preparation → explosive takeoff → suspended mid-air → quick landing",
            "floorwork with slow movement → fast jump → fast turn → held pose"
        ]
    }

    # 2. 优化后的舞蹈无级模板库（移除scene和direction相关内容）
    dance_templates = [
        "The dancer performed {combo_action} with {detail}, following {speed_rhythm}.",
        "During practice, the performer executed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The dancer completed {combo_action} with {detail} (dynamic control), adhering to {speed_rhythm}.",
        "In the sequence climax, the dancer executed {base_action} followed by {combo_action} with {detail}, using {speed_rhythm}.",
        "The performer combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The sequence began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the dancer practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In dynamic performance, the dancer executed {base_action} with {detail}, transitioning to {combo_action} with {speed_rhythm}."
    ]

    # 3. 计算最大可能的不重复组合数
    max_possible = 0
    for template in dance_templates:
        has_combo = "{combo_action}" in template
        has_base = "{base_action}" in template
        
        if has_combo and has_base:
            # 同时包含组合动作和基础动作
            max_possible += len(dance_vars["combo_action"]) * len(dance_vars["base_action"]) * len(dance_vars["detail"]) * len(dance_vars["speed_rhythm"])
        elif has_combo:
            # 仅包含组合动作
            max_possible += len(dance_vars["combo_action"]) * len(dance_vars["detail"]) * len(dance_vars["speed_rhythm"])
        elif has_base:
            # 仅包含基础动作
            max_possible += len(dance_vars["base_action"]) * len(dance_vars["detail"]) * len(dance_vars["speed_rhythm"])
    
    # 校验输入数量合理性
    if num_prompts > max_possible:
        raise ValueError(f"生成数量超过最大可能组合数（最大{max_possible}个）")
    if num_prompts <= 0:
        raise ValueError("生成数量必须为正整数")

    # 4. 随机生成不重复的prompt
    generated = set()
    while len(generated) < num_prompts:
        template = random.choice(dance_templates)
        
        vars_selected = {
            "detail": random.choice(dance_vars["detail"]),
            "speed_rhythm": random.choice(dance_vars["speed_rhythm"])
        }
        
        if "{combo_action}" in template:
            vars_selected["combo_action"] = random.choice(dance_vars["combo_action"])
        if "{base_action}" in template:
            vars_selected["base_action"] = random.choice(dance_vars["base_action"])
        
        prompt = template
        for key, value in vars_selected.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        generated.add(prompt)
    
    return list(generated)
def generate_random_combat_prompt(num_prompts):
    """
    Randomly generate a specified number of non-repetitive combat prompts (remove scene/direction, optimize variable format)
    Args:
        num_prompts: Number of prompts to generate (must not exceed the maximum possible combinations)
    Returns:
        List of non-repetitive combat prompts
    """
    # 1. Optimized combat variable library (remove numbers/brackets, delete scene & direction)
    combat_vars = {
        "base_action": [
            # Simple (Low-Dynamic)
            "basic jab with slow rhythm and small arm movement",
            "simple slip with small lateral shift and no sudden speed change",
            "basic footwork slide with steady pace and shoulder-width stance",
            "basic parry with minor arm adjustment and low force",
            "straight punch with no hip rotation and slow extension",
            "simple knee tap with low height and minimal core engagement",
            "static guard with arms up and no movement",
            "slow front kick with thigh-height movement and no follow-through",
            
            # Medium-Difficulty
            "standard cross with moderate hip rotation and steady speed",
            "basic roundhouse kick with small hip swing and flat foot landing",
            "basic full mount with static control and no rapid transition",
            "hook punch with bent arm and moderate torque",
            "side kick with mid-thigh height and controlled retraction",
            "half guard sweep with slow weight shift",
            "elbow strike with short range and moderate force",
            "rear naked choke with slow arm wrapping and no immediate pressure",
            "front headlock with static hold and no takedown attempt",
            
            # High-Dynamic/High-Difficulty
            "spinning back fist with rotational burst and fast arm swing",
            "flying knee with airborne explosion and core tension for balance",
            "jump switch kick with mid-air leg switch and hip alignment control",
            "armbar from closed guard with rapid joint lock and elbow pressure focus",
            "superman punch with body extension burst and airborne reach",
            "wheel kick with full leg rotation and wide swing",
            "triangle choke with rapid leg entanglement and neck pressure",
            "double-leg takedown with explosive drive and low center of gravity",
            # New high-dynamic actions
            "switch kick feint to spinning heel strike with feint and full rotation",
            "jumping switch knee with double-leg airborne movement and core locked for balance",
            "flying armbar with airborne diving movement, body folded mid-air and rapid joint control",
            "reverse spinning elbow with turning movement, rotation driven by shoulders and quick reset after impact",
            "leaping guillotine choke with forward diving movement, arms cinched instantly and body weight applied",
            "540° tornado kick with single-leg takeoff, leg arcing during rotation",
            "ankle lock from open guard with rapid lock and heel pressure plus body torsion",
            "diving double punch with forward diving movement, body leaned forward and quick transition to defensive stance",
            "spinning back kick to the body with turning movement, hip hyperextension and body rotation for reset",
            "cartwheel guard pass with cartwheel-style movement, instant arm support for force and crossed legs to pass guard"
        ],
        "combo_action": [
            # Simple Combos
            "basic jab → standard cross → standard retreat step",
            "simple parry → basic hook → simple slip",
            "straight punch → knee tap → footwork slide",
            "static guard → elbow strike → side step",
            
            # Medium-Difficulty Combos
            "hook punch → roundhouse kick → half guard transition",
            "side kick → cross punch → backward shuffle",
            "parry → elbow strike → forward lunge",
            "front headlock → hip toss → mount",
            
            # High-Dynamic Combos
            "simple slip → superman punch → flying knee → pivot escape",
            "cartwheel kick → landing slide → spinning back fist → weave",
            "double-leg takedown → mount → armbar → sweep",
            "wheel kick → landing spin → jump switch kick → defensive guard",
            "switch kick feint → spinning heel kick → diving double punch → retreat roll",
            "leaping guillotine → ankle lock → standup → 540° tornado kick"
        ],
        "detail": [
            # Basic Details
            "fist form with aligned knuckles and locked wrist during punch",
            "footwork with parallel feet and shoulder-width stance during slide",
            "guard position with tucked elbows and vertical forearms",
            "knee tap with flexed hip and flat foot before impact",
            "punch retraction with speed matching extension and no lag",
            
            # Medium-Difficulty Details
            "roundhouse kick with rotated supporting foot and raised knee first",
            "hook punch with elbow kept at right angle and force from torso twist",
            "half guard with legs wrapped above knee and applied hip pressure",
            "rear naked choke with forearm across windpipe and bicep pressed against jaw",
            
            # High-Difficulty Details
            "flying knee with tight core and body lean mid-air to maintain balance",
            "spinning back fist with eyes locked on target and heel-first landing for buffering",
            "superman punch with rear leg driving forward and extended torso",
            "triangle choke with crossed legs at ankles and elevated hip",
            "double-leg takedown with head pressed into opponent's chest and arms wrapped behind knees",
            # New high-difficulty action details
            "540° tornado kick with pivoted takeoff foot, arms swung for momentum and eyes fixed on target until landing",
            "flying armbar with body forming a 'C' shape mid-air, hips thrust immediately after arm lock and elbow joint kept perpendicular to ground",
            "jumping switch knee with knees tucked inward during leg switch, forefoot used for buffering on landing and center of gravity shifted to striking leg",
            "reverse spinning elbow with shoulders leading hip rotation during turn, elbow kept level with ears and arm tensed at impact"
        ],
        "speed_rhythm": [
            # Simple Rhythm
            "steady punching speed with consistent movement pace",
            "slow footwork with no sudden direction changes",
            "guard hold → slow punch → reset with balanced timing",
            
            # Medium-Difficulty Rhythm
            "alternating speed: slow jab → medium cross → medium hook",
            "slow step → moderate kick → slow retreat",
            "stance hold → strike → recovery with steady timing",
            
            # High-Dynamic Rhythm
            "slow slide → explosive punch → immediate stop post-impact",
            "slow pre-spin windup → speed burst during spinning back fist → fast defensive reset",
            "takedown drive with acceleration → sudden weight shift → submission lock",
            "slow feint → fast real strike → rapid direction change",
            "airborne movement with takeoff → mid-air posture control → landing buffer → quick transition to next move"
        ]
    }

    # 2. Optimized combat template library (remove scene & direction)
    combat_templates = [
        "The fighter executed {combo_action} with {detail}, following {speed_rhythm}.",
        "During training, the combatant performed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The fighter completed {combo_action} at {speed_rhythm}, with {detail} (precision control).",
        "The combatant launched {base_action} with {detail}, maintaining {speed_rhythm}.",
        "The fighter combined {base_action} and {combo_action}, using {detail} to optimize {speed_rhythm}.",
        "The combatant transitioned from {combo_action} to {base_action} with {detail}, guiding {speed_rhythm}."
    ]

    # 3. Calculate maximum possible non-repetitive combinations
    max_possible = 0
    for template in combat_templates:
        has_combo = "{combo_action}" in template
        has_base = "{base_action}" in template
        
        if has_combo and has_base:
            max_possible += len(combat_vars["combo_action"]) * len(combat_vars["base_action"]) * len(combat_vars["detail"]) * len(combat_vars["speed_rhythm"])
        elif has_combo:
            max_possible += len(combat_vars["combo_action"]) * len(combat_vars["detail"]) * len(combat_vars["speed_rhythm"])
        elif has_base:
            max_possible += len(combat_vars["base_action"]) * len(combat_vars["detail"]) * len(combat_vars["speed_rhythm"])
    
    # Validate input quantity
    if num_prompts > max_possible:
        raise ValueError(f"Number of prompts exceeds maximum possible combinations (max: {max_possible})")
    if num_prompts <= 0:
        raise ValueError("Number of prompts must be a positive integer")

    # 4. Randomly generate non-repetitive prompts
    generated = set()
    while len(generated) < num_prompts:
        template = random.choice(combat_templates)
        vars_selected = {
            "detail": random.choice(combat_vars["detail"]),
            "speed_rhythm": random.choice(combat_vars["speed_rhythm"])
        }
        
        if "{combo_action}" in template:
            vars_selected["combo_action"] = random.choice(combat_vars["combo_action"])
        if "{base_action}" in template:
            vars_selected["base_action"] = random.choice(combat_vars["base_action"])
        
        prompt = template
        for key, value in vars_selected.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        generated.add(prompt)
    
    return list(generated)

PROMPT_GENERATORS = {
    "sport": generate_random_sports_prompt,
    "gymnastics": generate_random_gymnastics_prompt,
    "martial_arts": generate_random_martial_arts_prompt,
    "dance": generate_random_dance_prompt,
    "combat": generate_random_combat_prompt,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate loop0 prompts for CLAIMS.")
    parser.add_argument(
        "--output_dir",
        default="/home/group16/xuws/MDM_DIP/CLAIMS/prompt-generate/prompts/random_loop0_prompts_motions",
        help="Directory to store per-category prompt files and the merged loop0_prompts.txt file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Number of prompts to generate per category unless overridden by a category-specific argument.",
    )
    parser.add_argument("--sport_count", type=int, default=None, help="Number of sport prompts.")
    parser.add_argument("--gymnastics_count", type=int, default=None, help="Number of gymnastics prompts.")
    parser.add_argument("--martial_arts_count", type=int, default=None, help="Number of martial arts prompts.")
    parser.add_argument("--dance_count", type=int, default=None, help="Number of dance prompts.")
    parser.add_argument("--combat_count", type=int, default=None, help="Number of combat prompts.")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(PROMPT_GENERATORS.keys()),
        default=list(PROMPT_GENERATORS.keys()),
        help="Categories to generate. Defaults to all categories.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    return parser.parse_args()


def write_prompt_file(file_path, prompts):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
        f.write("\n")


def normalize_prompt_text(text):
    normalized = text.lower().replace("_", " ").replace("→", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def write_prompt_manifest(file_path, prompt_records):
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_name", "category", "normalized_prompt"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for record in prompt_records:
            writer.writerow({
                "prompt_name": record["prompt_name"],
                "category": record["category"],
                "normalized_prompt": normalize_prompt_text(record["prompt_name"]),
            })


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    category_counts = {
        "sport": args.sport_count if args.sport_count is not None else args.count,
        "gymnastics": args.gymnastics_count if args.gymnastics_count is not None else args.count,
        "martial_arts": args.martial_arts_count if args.martial_arts_count is not None else args.count,
        "dance": args.dance_count if args.dance_count is not None else args.count,
        "combat": args.combat_count if args.combat_count is not None else args.count,
    }

    merged_prompts = []
    prompt_records = []

    for category in args.categories:
        generator = PROMPT_GENERATORS[category]
        prompts = generator(category_counts[category])
        file_path = os.path.join(args.output_dir, f"{category}_prompt.txt")
        write_prompt_file(file_path, prompts)
        merged_prompts.extend(prompts)
        prompt_records.extend(
            {"prompt_name": prompt, "category": category}
            for prompt in prompts
        )
        print(f"[{category}] generated {len(prompts)} prompts -> {file_path}")

    merged_file_path = os.path.join(args.output_dir, "loop0_prompts.txt")
    write_prompt_file(merged_file_path, merged_prompts)
    print(f"[merged] wrote {len(merged_prompts)} prompts -> {merged_file_path}")

    manifest_path = os.path.join(args.output_dir, "loop0_prompt_manifest.csv")
    write_prompt_manifest(manifest_path, prompt_records)
    print(f"[manifest] wrote {len(prompt_records)} prompt-category rows -> {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"错误: {e}")


