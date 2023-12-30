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
[![Issues][issues-shield]][issues-url]
<!-- [![MIT License][license-shield]][license-url] -->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- ABOUT THE PROJECT -->
## What does this do?

It rectifies the [token boundary bias](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) in greedy tokenization.
Example: Given a prompt with a partial url ending with `:`. The model might have seen the desired `://` as a single token in training, but seeing just `:` tells it: "the next token is likely not `//`, otherwise the token would've been `://`".
This includes the problem of output sensitivity to prompts ending with whitespace or punctuation.

<!-- REFERENCES -->
## References

https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38

<!-- GETTING STARTED -->
## Getting Started

### Installation

`pip install .` should pick up `[tool.poetry.dependencies]` from `pyproject.toml`, which are `transformers` and `pygtrie`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

```py
from token_healing import TokenBoundaryHealer

token_healer = TokenBoundaryHealer(model, tokenizer)
query = 'The link is <a href="http:'
healed_query = token_healer(query)
# The link is <a href="http://
```

See `example.py` for more detailed usage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Write a roadmap -->

<!-- See the [open issues](https://github.com/Ayenem/TokenHealerissues) for a full list of proposed features (and known issues). -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



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
