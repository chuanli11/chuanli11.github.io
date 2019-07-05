---
layout: post
title:  "About This Website"
date:   2019-07-04 10:00:00
categories: jekyll
---

This website is made with [Jekyll](https://jekyllrb.com/), a middle man between your favorite markup language and a extendable, beautiful static website.

### Basic Setup

I started by hosting on Github a basic website using Jekyll's default minima theme.

#### Step 1: Create a Github host repo

Create a repo with the name `username.github.io`. This is going to be the [`url`](https://chuanli11.github.io/) to access the website.

#### Step 2: Clone the rep

```
git clone https://github.com/username/username.github.io.git
```
#### Step 3: Install Dependency for Jekyll

I follow the instructions [here](https://jekyllrb.com/docs/installation/ubuntu/):

```
sudo apt-get install ruby-full build-essential zlib1g-dev
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 4: Create a basic Jekyll project

```
cd username.github.io

jekyll new . --force

# Check jekyll version in Gemfile
gem "jekyll", "~> 3.8.5"

# Add github-pages gem
gem "github-pages", group: :jekyll_plugins

bundle update

gem install bundler jekyll
```

#### Step 5: Test 

```
bundle exec jekyll serve

# Go to http://127.0.0.1:4000/
```

#### Step 6: Push

```
# Add this to .gitignore
_site/

git add -A
git commit -m "First commit"
git push origin mater
```
Now you can see the website alive at `https://username.github.io/`


### Customize
Having the basic website running alive, I then customize it for my own need.


#### Add Credential
Fill in some basic information in `_config.yml` and `about.md`

#### Add New Post
Creat this blog to walk through the step of creating this website. Remove the welcome post created by Jekyll.

#### Add Pop Footnote

#### Add Sticky Table of Content

