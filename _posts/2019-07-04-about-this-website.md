---
layout: post
title:  "About This Website"
date:   2019-07-04 10:00:00
categories: jekyll
---

**Contents**
* TOC
{:toc}

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
I use the [Bigfoot](http://www.bigfootjs.com/) jQuery plugin to enable popup footnote. This is extremely helpful for writing long articles as one does not need to jump between the content and the footnote[^BigFoot].

The instructions are copied from this [blog](https://sherif.io/2014/11/07/Bigfoot-in-Jekyll.html):

Download [Bigfoot](http://www.bigfootjs.com/)

Download [jQuery](https://code.jquery.com/jquery-3.4.1.min.js) 

Create a `js` folder in the root directory. Copy `jquery-3.4.1.js` and `bigfoot.js` there. Set `useFootnoteOnlyOnce: false` in `bigfoot.js` to enable bigfoot on multiple references of the same footnote.


Create a `css` folder in the root directory. Add a `style.scss` file with the following content:

```
---
# This is the main default style sheet
---
@import "main";
```

Create a `_sass` folder in the root directory. Copy the `bigfoot-number.scss` file into it. Also add a `main.scss` file with a following content:

```
@import "bigfoot-number";
```

Create a `_layout` folder in the root directory. Add [`default.html`](https://github.com/chuanli11/chuanli11.github.io/blob/master/_layouts/default.html) file.



#### Add Table of Content

Unfortunately, Github Pages only have limited support for Jekyll gems. For this reason, the `jeklyy-toc` gem only works locally. Once pushed to Github page, the table of content will disappear. 

Fortunately, rendering of table of content is supported by kramdown, the current default Markdown processor and the only supported Markdown processor on Githug Pages. Adding the following code snippet to each post will create a table of content:

```
---
front matter
---

* TOC
{:toc}
```
#### Reference

[^BigFoot]: This is how a nice popup footnote looks like!