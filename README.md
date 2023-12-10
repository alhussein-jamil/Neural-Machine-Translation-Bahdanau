# MLA_Project
Neural Machine Translation by Jointly Learning to Align and Translate paper implementation 
https://arxiv.org/abs/1409.0473v7

## Requirements
installation 
```
pip install --upgrade -r requirements.txt
pip install -e ./src
``` 


## Pulling from main branch
```
git stash 
git pull origin main
git stash pop
```

## Changing branch
```
git checkout <branch_name>
```

## Pushing to branch
```
git add .
git commit -m "message"
git push origin <branch_name>
```

## Creating a pull request
```
git checkout main
git pull origin main
git checkout <branch_name>
git merge main
git push origin <branch_name>
```
Then go to github and create a pull request
