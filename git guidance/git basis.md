# Basis

## Workspace

<img src="./img/workspace.jpg">

- 工作区（Working Directory）：文件夹所在地方，存放各种文件及文件夹

- 版本库（Repository）：工作区有一个隐藏目录`.git`，是Git的版本库。Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支`master`，以及指向`master`的一个指针叫`HEAD`。提交到版本库时有两个步骤：

  - `git add`把文件添加进去，实际上就是把文件修改添加到暂存区；

  - `git commit`提交更改，实际上就是把暂存区的所有内容提交到当前分支。

  **注意**：Git管理的是修改，`git add`是将修改放入暂存区，准备提交，`git commit`提交暂存区的修改。

- GitHub远程仓库

## Command

本地操作

`git log` / `git log --pretty=oneline`

​	用于查看历史提交版本，其中Git中的`HEAD`表示提交的版本id。Git必须知道当前版本是哪个版本，在Git中，用`HEAD`表示当前版本，也就是最新的提交`1094adb...`（注意我的提交ID和你的肯定不一样），上一个版本就是`HEAD^`，上上一个版本就是`HEAD^^`，当然往上100个版本写100个`^`比较容易数不过来，所以写成`HEAD~100`

`git relog`

​	查看回退前的历史提交版本，以方便通过id回滚找回历史提交。

`git reset --hard [branch id]`

​	回退到某一个版本上

​	**其中**，`git reset --hard HEAD`相当于做`revert`到修改前。

`git status`

​	查看状态，显示目前在工作区的文件状态，被修改了会显示`modified`，从来没有被添加过状态是`Untracked`等。

`git add [filename]` /`git add -A` 

​	将修改提交到暂存区

`git commit -m "[notes]"`

​	将修改提交到版本库中，备注相应的解释说明

`git checkout -- file` 用于`add`之前，文件还没被放到暂存区

​	可以丢弃工作区的修改，文件在工作区的修改全部撤销，这里有两种情况：一种是`readme.txt`自修改后还没有被放到暂存区，撤销修改就回到和版本库一模一样的状态；一种是`readme.txt`已经添加到暂存区后，又作了修改，撤销修改就回到添加到暂存区后的状态。总之，就是让文件回到最近一次`git commit`或`git add`时的状态。

**！！**其实是用版本库里的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。

`git reset HEAD <file>` 用于`commit`之前

​	可以把暂存区的修改撤销掉，重新放回工作区。`git reset`命令既可以回退版本，也可以把暂存区的修改回退到工作区。当我们用`HEAD`时，表示最新的版本。

`git rm`

​	用于删除一个文件。如果一个文件已经被提交到版本库，不用担心误删，但是要小心，因为只能恢复文件到最新版本，你会丢失**最近一次提交后你修改的内容**。

### 远程仓库操作

`git clone [ssh/http]`

​	从远程仓库克隆到本地

`git push`

​	将本地提交记录传到远程仓库

`git pull`

​	将远程仓库的更新传到本地工作区