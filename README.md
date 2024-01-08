<!-- back to top link -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url] -->
<!-- [![Forks][forks-shield]][forks-url] -->
<!-- [![Stargazers][stars-shield]][stars-url] -->
<!-- [![Issues][issues-shield]][issues-url] -->
<!-- [![MIT License][license-shield]][license-url] -->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->
[![Lines of Code](https://tokei.rs/b1/github/Ayenem/TokenHealer?category=code)](https://github.com/Ayenem/TokenHealer)



<!-- ABOUT THE PROJECT -->
## What is token healing?

Token healing rectifies the token boundary bias in greedy tokenization. It does this by trimming and regrowing the prompt to better align with the model's tokenizer, thus enhancing generation quality. The improvement is clearest with completion models.

Example: given a completion prompt with a partial url ending with `:`, the model might have seen the expected completion `://` as a _single_ token in training. However, the prompt's tail token `:` tells it that the next token is not `//`, and so it looks for wrong completions. Such errors compound in auto-regressive language models.

Debiasing token boundaries also addresses output sensitivity to prompts ending with whitespace.

A more thorough explanation can be found on [The Art of Prompt Design: Prompt Boundaries and Token Healing | by Scott Lundberg](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38).

<!-- REFERENCES -->
<!-- ## References -->


<!-- GETTING STARTED -->
<!-- ## Getting Started -->

## Installation

`pip install .` should pick up the main dependencies from `pyproject.toml`, that is, `transformers` and `pygtrie`. You could also just copy-paste `token_healing.py` and install as needed.

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- USAGE EXAMPLES -->
## Usage

```py
from token_healing import TokenBoundaryHealer

prompt = 'The link is <a href="http:'

output = generate(prompt, completion_model, tokenizer)
# The link is <a href="http:&#47;&#47;www&#47;dailymail&#

# The model saw '://' as a single token in training. Seeing a prompt ending with `:` tells it that the
# next token is likely not `//`, because otherwise it would've seen `://`.
# Thus, it completes with a token other than `//`, in this case, `&`.

token_healer = TokenBoundaryHealer(completion_model, tokenizer)
healed_prompt = token_healer(prompt)
# The link is <a href="http://
healed_output = generate(healed_prompt, completion_model, tokenizer)
# The link is <a href="https://www.365doki.com/post/3699
```

See `example.py` for the full example.

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Write a roadmap -->

<!-- See the [open issues](https://github.com/Ayenem/TokenHealerissues) for a full list of proposed features (and known issues). -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information. -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- CONTACT -->
## Contact

Ahmed Moubtahij - [@TheAyenem](https://twitter.com/TheAyenem)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/Ayenem/TokenHealing.svg?style=for-the-badge
[contributors-url]: https://github.com/Ayenem/TokenHealergraphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Ayenem/TokenHealing.svg?style=for-the-badge
[forks-url]: https://github.com/Ayenem/TokenHealernetwork/members
[stars-shield]: https://img.shields.io/github/stars/Ayenem/TokenHealing.svg?style=for-the-badge
[stars-url]: https://github.com/Ayenem/TokenHealerstargazers
[issues-shield]: https://img.shields.io/github/issues/Ayenem/TokenHealing.svg?style=for-the-badge
[issues-url]: https://github.com/Ayenem/TokenHealerissues
[license-shield]: https://img.shields.io/github/license/Ayenem/TokenHealing?style=for-the-badge
[license-url]: https://github.com/Ayenem/TokenHealerblob/main/LICENSE.txt
-->
