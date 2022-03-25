# find some file
find . -regex '.*\.txt\|.*\.doc\|.*\.mp3'  

# except some file
find . -type f ! -name "*.html"   

# skip the file scp
rsync -avzu --progress -e "ssh -p22" root@xx.xx.xx:/xx/xx/xx ./

# find the files more than 100M
find ./ -type f -size +100M

# git name

git config --global user.name
git config --global user.email 