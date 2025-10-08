# Git Basics

## Cloning a Repository
**What it does:** this copies the files on a remote server (in this case github) to your computer

```bash
git clone https://github.com/supermepsipax/med-eng-signal-project.git
```

## Creating a New Branch
**What it does:** Creates a separate workspace where you can make changes without affecting the main code. I think just making a branch with your name will be the easiest

```bash
git branch <branch-name>


```

## Switching Branches
**What it does:** Moves you to a different branch so you can work on it.

```bash
git checkout <branch-name>
```
You can also see which branch you are on using, you'll see a message
```bash
git status
```

**Shortcut:** Create and switch to a new branch in one command:
```bash
git checkout -b <branch-name>
```

## Adding and Committing Changes
**What it does:** Saves your changes to Git's history.

1. **Add files** (stages them for commit):
```bash
git add <file-name>          # Add a specific file
git add .                    # Add all changed files
```

2. **Commit** (saves the staged changes with a message):
```bash
git commit -m "Your descriptive message here"
```

Example:
```bash
git add .
git commit -m "Add login form validation"
```

## Pushing Changes
**What it does:** Uploads your local commits to the remote server so others can see them.

```bash
git push origin <branch-name>
```

**First time pushing a new branch:**
```bash
git push -u origin <branch-name>
```

## Pulling Changes
**What it does:** Downloads the latest changes from the remote server and merges them into your current branch.

```bash
git pull
```

This keeps your local code up-to-date with what others have pushed.

## Storing credentials

If you run into problems with logging in you will need to make an access token, go to your github account, click on your user icon, go to setting - developer settings - personal access tokens - generate new token (classic). Create a new token and set it to an expiry date that is longer than the class. It will display the token only once so make sure you save it somewhere securely. 

Before you try doing an operation setup you name and email like this.

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
And then after that configure your credential storing according to your system below.

# On Windows
```bash
git config --global credential.helper wincred
```
# On macOS
```bash
git config --global credential.helper osxkeychain
```

# On Linux
```bash
git config --global credential.helper store
```
Once this is done, when you are prompted for you username and password, enter your actual github username but don't use your password, use the access token you generated and saved previously. This credential should be stored so you don't have to enter it again.

## Creating a Merge Request (Pull Request)
**What it does:** Asks to merge your branch into the main branch, allowing others to review your code first.

This is typically done through the web interface (GitHub, GitLab, etc.):

1. Push your branch to the remote repository
2. Go to your repository on the website
3. Click "New Pull Request" or "New Merge Request"
4. Select your branch as the source and `main` as the target
5. Add a title and description of your changes
6. Submit the request for review

Once approved, your changes will be merged into the main branch.


## Common Workflow Summary
```bash
# 1. Clone the repository (first time only)
git clone <repository-url>

# 2. Create and switch to a new branch
git checkout -b feature-name

# 3. Make your changes to the code

# 4. Add and commit your changes
git add .
git commit -m "Description of changes"

# 5. Push your branch to the remote
git push -u origin feature-name

# 6. Create a merge request on the website

# 7. After merge, switch back to main and update
git checkout main
git pull
```

## Helpful Commands
- `git status` - See what files have changed
- `git log` - View commit history
- `git branch` - List all branches (current branch marked with *)
- `git diff` - See what changes you've made
