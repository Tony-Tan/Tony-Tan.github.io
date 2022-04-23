#!/bin/bash


git add .

# 设置提交说明，格式为 Site updated: 2006-01-02 15:04:05
time=$(date "+%Y-%m-%d %H:%M:%S")
commit="Site updated:"$time
echo $commit

# 提交
git commit -m "$commit"

# 推送到source分支上
git push origin source
