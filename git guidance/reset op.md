- #### 恢复 `git reset -hard` 的误操作

  `reset`, `rebase` 和 `merge`是Git上需要谨慎处理的操作。`git reset --hard [id]` 是将版本恢复到对应的版本号中。找回恢复之前的版本可以采用下面的操作。

  - `git reflog` 查看需要恢复的版本号。
  - `git reset --hard [id]` 恢复到该版本中。

  `Git` 中的`-soft`, `-mixed`, `-hard` ：

  - `HEAD`
    这是当前分支版本顶端的别名，也就是在当前分支你最近的一个提交，也就是本地仓库，即你的`commit`记录

  - `Index`
    `index`也被称为`staging area`，即`add`的记录

  - `Working Copy`
    `working copy`代表你正在工作的那个文件

  `-soft` : `HEAD` != `index` = `Working Copy`**只撤销了commit ，保留了index（add过）和工作区**

  `-mixed` : `HEAD` = `index` ！= `Working Copy`**撤销了commit 、index，工作区不变**

  `-hard` : `HEAD` = `index` = `Working Copy`**commit 、index和工作区文件都回退改变**

