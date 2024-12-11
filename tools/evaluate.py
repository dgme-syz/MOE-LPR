

# qwen pretrain template
# From llama factory
def Formatter(slots):
    def apply(self, **kwargs):
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements
        

class QwenTemplate:
    format_user=Formatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"])
    format_system=Formatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"])
    format_separator=Formatter(slots=["\n"])
    default_system="You are a helpful assistant."
    stop_words=["<|im_end|>"]
    replace_eos=True
    
    # given a list of messages, encode them into a prompt_id and answer_id
    def encode(self, tokenizer, messages, system):
        system = system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0 and system:
                elements += self.format_system.apply(content=system)
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == "user":
                elements += self.format_user.apply(content=message["content"])
            elif message["role"] == "assistant":
                elements += self.format_assistant.apply(content=message["content"])
            else:
                raise NotImplementedError

            for elem in elements:
                encoded_elem = tokenizer.encode(elem, add_special_tokens=False)
                encoded_messages += encoded_elem
        encoded_pairs = []
        for i in range(0, len(encoded_messages), 2):
            encoded_pairs.append((encoded_messages[i], encoded_messages[i + 1]))
        prompt_ids = []
        for q, r in encoded_pairs[:-1]:
            prompt_ids += q + r
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids
        
CHOICES = ["A", "B", "C", "D"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]


class EvalTemplate:
    system=Formatter(slots=["The following are multiple choice questions (with answers) about {subject}.\n\n"])
    choice=Formatter(slots=["{choice}. {content}\n"])
    answer=Formatter(slots=["\n\nAnswer: "])
    prefix=" "

    def parse_example(self, example):
        candidates = [self.choice.apply(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer.apply("")]), example["answer"]

    def format_example(self, target_data, support_set, subject_name):
        messages = []
        for k in range(len(support_set)):
            prompt, response = self.parse_example(support_set[k])
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})

        prompt, response = self.parse_example(target_data)
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        messages[0]["content"] = self.system.apply(subject=subject_name) + messages[0]["content"]
        return messages
    
    
    
    

