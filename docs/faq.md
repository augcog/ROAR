# FAQ
1. I did `git pull origin main`, no updates were made, why?

Remember that we are using submodules. If you are trying to update `ROAR_Sim`/`ROAR_Jetson`/`ROAR_Gym`, please understand that they are submodules, 
and refer to this [stackoverflow post](https://stackoverflow.com/questions/4611512/is-there-a-way-to-make-git-pull-automatically-update-submodules)

Also check if `origin` is actually pointing to `http://www.github.com/YOURUSERNAME` if you intended to update your own github
Please checkout this document to understand git remotes: [https://www.atlassian.com/git/tutorials/syncing](https://www.atlassian.com/git/tutorials/syncing)

