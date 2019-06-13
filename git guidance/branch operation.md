# Branch

## What is Branch？

​	分支（branch）相当于创建了一个属于你自己的分支，别人看不到，还继续在原来的分支上正常工作，而在自己的分支上干活，想提交就提交，直到开发完毕后，再一次性合并到原来的分支上，这样，既安全，又不影响别人工作。每次提交，Git都把它们串成一条时间线，这条时间线就是一个分支。截止到目前，只有一条时间线，在Git里，这个分支叫主分支，即`master`分支。`HEAD`严格来说不是指向提交，而是指向`master`，`master`才是指向提交的，所以，`HEAD`指向的就是当前分支。

​	一开始的时候，`master`分支是一条线，Git用`master`指向最新的提交，再用`HEAD`指向`master`，就能确定当前分支，以及当前分支的提交点：

![0](.\img\0.png)



​	新建分支`dev`

![0 (2)](.\img\0 (2).png)

​	在分支上进行`commit`操作等。

<div align=center><img src=".\img\0 (1).png"></div>

​	合并分支`dev`

<div align=center><img src=".\img\0 (3).png"></div>

## 分支操作

### Basis

- 查看分支：`git branch`：`git branch`命令会列出所有分支，当前分支前面会标一个`*`号。
- 创建分支：`git branch <name>`
- 切换分支：`git checkout <name>`
- 创建+切换分支：`git checkout -b <name>`
- 合并某分支到当前分支：`git merge <name>`
- 禁止快速合并分支：`git merge --no-ff -m "merge with no-ff" <name>`
- 删除分支：`git branch -d <name>`
- 强行删除一个没有被合并过的分支：`git branch -D <name>`

### 冲突解决

​	通常情况下，如果`dev`和`master`不产生冲突的话，合并会是`fast-forward`模式。但是如果`dev`和`master`修改中出现冲突，会导致分支无法合并。采用`git status`也可以告诉我们冲突的文件。查看冲突文件，Git会用`<<<<<<<`，`=======`，`>>>>>>>`标记出不同分支的内容，根据标记内容进行修改后，保存并再次提交。

​	用带参数的`git log`查看分支的合并情况，如：`git log --graph --pretty=oneline --abbrev-commit`

​	如果合并分支时，加上`--no-ff`参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而`fast forward`合并就看不出来曾经做过合并。

​	差别：

<div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\0 (6).png" height="190" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\0 (7).png" height="190" ></div>

<div style="float:left;border:solid 1px 000;margin-left:150px;">fast-forward模式</div><div style="float:left;border:solid 1px 000;margin-left:300px;">--no-ff模式</div>

### 分支策略

在实际开发中，我们应该按照几个基本原则进行分支管理：

首先，`master`分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；

那在哪干活呢？干活都在`dev`分支上，也就是说，`dev`分支是不稳定的，到某个时候，比如1.0版本发布时，再把`dev`分支合并到`master`上，在`master`分支发布1.0版本；

你和你的小伙伴们每个人都在`dev`分支上干活，每个人都有自己的分支，时不时地往`dev`分支上合并就可以了。

所以，团队合作的分支看起来就像这样：

<div align=center><img src=".\img\0 (5).png"></div>

### bug 修复

​	修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；当手头工作没有完成时，先把工作现场`git stash`一下，然后去修复bug，修复后，再`git stash pop`，回到工作现场。

1、`stash`功能，可以把当前工作现场“储藏”起来，等以后恢复现场后继续工作

```
$ git stash
Saved working directory and index state WIP on dev: f52c633 add merge
```

2、修复bug后，用`git stash list`命令看看：

```
$ git stash list
stash@{0}: WIP on dev: f52c633 add merge
```

3、工作现场还在，Git把stash内容存在某个地方了，但是需要恢复一下，有两个办法：

一是用`git stash apply`恢复，但是恢复后，stash内容并不删除，你需要用`git stash drop`来删除；

另一种方式是用`git stash pop`，恢复的同时把stash内容也删了：

```
$ git stash pop
On branch dev
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

Dropped refs/stash@{0} (5d677e2ee266f39ea296182fb2354265b91b3b2a)
```

4、你可以多次stash，恢复的时候，先用`git stash list`查看，然后恢复指定的stash，用命令：

```
$ git stash apply stash@{0}
```

### 远程分支提交

- 查看远程库信息，使用`git remote -v`；
- 本地新建的分支如果不推送到远程，对其他人就是不可见的；
- 从本地推送分支，使用`git push origin branch-name`，如果推送失败，先用`git pull`抓取远程的新提交；
- 在本地创建和远程分支对应的分支，使用`git checkout -b branch-name origin/branch-name`，本地和远程分支的名称最好一致；
- 建立本地分支和远程分支的关联，使用`git branch --set-upstream branch-name origin/branch-name`；
- 从远程抓取分支，使用`git pull`，如果有冲突，要先处理冲突。

- `git rebase`操作可以把本地未push的分叉提交历史整理成直线；目的是使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比。

### 标签

- 命令`git tag <tagname>`用于新建一个标签，默认为`HEAD`，也可以指定一个commit id；
- 命令`git tag -a <tagname> -m "blablabla..."`可以指定标签信息；
- 命令`git tag`可以查看所有标签。
- 命令`git push origin <tagname>`可以推送一个本地标签；
- 命令`git push origin --tags`可以推送全部未推送过的本地标签；
- 命令`git tag -d <tagname>`可以删除一个本地标签；
- 命令`git push origin :refs/tags/<tagname>`可以删除一个远程标签。