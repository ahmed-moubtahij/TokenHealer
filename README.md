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



<!-- ABOUT THE PROJECT -->
## What does this do?

It rectifies the [token boundary bias](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) in greedy tokenization.

Example: given a prompt with a partial url ending with `:`. The model might have seen the desired `://` as a single token in training, but seeing just `:` tells it that the next token is likely not `//`, because otherwise it would've seen `://`. As one can imagine, such errors compound in auto-regressive language models.

Debiasing token boundaries also addresses output sensitivity to prompts ending with whitespace.

<!-- REFERENCES -->
<!-- ## References -->


<!-- GETTING STARTED -->
<!-- ## Getting Started -->

## Installation

`pip install .` should pick up the main dependencies from `pyproject.toml`, that is, `transformers="^4.36.2"` and `pygtrie="^2.5.0"`. You could also just copy-paste `token_healing.py` and install as needed.

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- USAGE EXAMPLES -->
## Usage

```py
from token_healing import TokenBoundaryHealer

prompt = 'The link is <a href="http:'

output = generate(prompt, model, tokenizer)
# The link is <a href="http:&#47;&#47;www&#47;dailymail&#

# The model saw '://' as a single token in training. Seeing a prompt ending with `:` tells it that the next token is likely not `//`, because otherwise it would've seen `://`. Thus, it completes with a token other than `//`, in this case, `&`.

token_healer = TokenBoundaryHealer(model, tokenizer)
healed_prompt = token_healer(prompt)
# The link is <a href="https://
healed_output = generate(healed_prompt, model, tokenizer)
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
