# 🎴 TCGP Team Rocket Tool

<div align="center">
  <img src="gui/icon.ico" alt="TCGP Team Rocket Tool Logo" width="150"/>
  
  **Complete TCG Pocket Collection Manager & Discord Bot**
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
  [![Discord.py](https://img.shields.io/badge/Discord.py-2.3+-blueviolet.svg)](https://pypi.org/project/discord.py/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## 🚀 Project Description

The TCGP Team Rocket Tool is a comprehensive application designed for managing your Trading Card Game (TCG) Pocket collection. It consists of a desktop application built with PyQt5 for local collection management and a Discord bot using Discord.py to integrate collection data with your Discord server.

This tool allows you to:

-   Maintain a detailed inventory of your TCG collection.
-   Quickly search and filter cards.
-   Export and import collection data.
-   Integrate with Discord for sharing and trading.

---

## ⚙️ Installation

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package installer)

### Steps

1.  **Clone the repository:**

    bash
    python main.py
    -   **Add Cards:** Manually add cards to your collection with details like set, condition, and quantity.
-   **Search & Filter:** Quickly find cards using various filters.
-   **Export/Import:** Export your collection to a file for backup or import from a file.

### Troubleshooting

-   **Application does not start:**
    -   Ensure all dependencies are installed correctly.
    -   Check for any error messages in the console.
    -   Verify that you are using a compatible version of Python.

---

## 🤖 Discord Bot Usage

### Setting up the Bot

1.  **Create a Discord Bot:**
    -   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    -   Create a new application and then create a bot within that application.
    -   Note your bot's token.

2.  **Invite the Bot to your Server:**
    -   Use the OAuth2 URL Generator in the Discord Developer Portal to generate an invite link with the `bot` and `applications.commands` scopes.
    -   Use the generated link to invite the bot to your server.

3.  **Configure the Bot:**
    -   Rename the `.env.example` file to `.env`
    -   Edit the `.env` file and add your bot token and any other required configurations:

        ### Bot Commands

> Update the following commands based on your bot's actual commands.

-   `/collection add <card_name> <set_name> <quantity>`: Adds a card to your collection.
-   `/collection remove <card_name> <set_name> <quantity>`: Removes a card from your collection.
-   `/collection view`: Displays your current collection.

### Troubleshooting

-   **Bot is not online:**
    -   Ensure the bot token is correct in the `.env` file.
    -   Check the console for any error messages.
    -   Verify that the bot has the necessary permissions in your Discord server.
-   **Commands are not working:**
    -   Make sure the bot is invited with the `applications.commands` scope.
    -   Ensure that the bot has the necessary permissions in the channel.

---

## 🤝 Contributing

We welcome contributions! Here's how you can contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive commit messages.
4.  Submit a pull request.

> Please follow these guidelines when contributing:
>
> -   Write clear and concise code.
> -   Add comments to explain complex logic.
> -   Test your changes thoroughly.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🐛 Bug Reports & Feature Requests

Found a bug or have a feature request? Please open an issue on the [GitHub Issues](https://github.com/yourusername/TCGP-TeamRocket-Tool/issues) page.

> When reporting bugs, please include:
>
> -   A clear and descriptive title.
> -   Steps to reproduce the bug.
> -   The expected behavior.
> -   The actual behavior.
> -   Any relevant error messages or screenshots.

---

## 💬 Community & Support

Join our Discord server for support and discussions:

[![Discord](label=Discord&logo=discord&logoColor=white)](https://discord.gg/Msa5vNjUUf)

---