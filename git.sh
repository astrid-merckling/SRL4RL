
set -e # stops the execution of a script if a command or pipeline has an error
set -x # all executed commands are printed to the terminal

git pull
git add --all
git commit -m "$@"
git push origin
