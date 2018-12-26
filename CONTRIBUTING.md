Contributing to DeepLearningZeroToAll
‘모두가 만드는 모두를 위한 딥러닝’ 참여 방법!! (Contribution)
Precheck steps : 사전확인
Before starting to work on something, please leave an issue first. Because
• 작업을 시작하기 전에 먼저 이슈를 남겨 두세요. 왜냐면
It helps to let people know what you are working on
당신이 무엇을 하고 있는지 사람들에게 알리는 데 도움이 됩니다.
A problem might have nothing to do with this repo
문제는 이 레포지토리와 무관할 수 있습니다
It could be our intention to keep the code in that way (for KISS)
그런 방식으로 코드를 유지하는 게 우리의 의도일 수도 있습니다
You should know how to use git
여러분은 git을 어떻게 사용하는지 알아야합니다.
If not, please google "how to use git." and read through them before you do anything. It's a mandatory skill to survive as a developer
그렇지 않다면, "git 사용 방법"을 검색한 후, 무언가를 하기 전에 그것들을 읽어 보세요. 개발자로서 살아남기 위해서는 필수적인 기술입니다.
Try git tutorial
Git tutorial을 참고하세요
Contribution guidelines
This document will guide you through the contribution process.
이 문서는 기여 프로세스를 안내합니다.
Step 1: Fork
Fork the project on GitHub by pressing the "fork" button. This step will copy this repository to your account so that you can start working on.
Fork 버튼을 눌러 GitHub에 프로젝트를 Fork하세요. 이 단계는 작업을 시작할 수 있게 당신의 계정에 복사하게 됩니다.
Step 2: Download to a local computer
Local computer에 다운로드하세요
$ git clone https://github.com/`YOUR_GITHUB_NAME`/deeplearningzerotoall/TensorFlow.git <- 주소 바꿔야 할듯
$ cd deeplearningzerotoall/TensorFlow <- 디렉토리명 바꿔야 할듯

Step 3: Setup an upstream
It's always a good idea to set up a link to this repo so that you can pull easily if there were changes
변경 사항이 있을 경우, 쉽게 pull할 수 있도록 이 repo에 대한 링크를 설정해야합니다.
$ git remote add upstream https://github.com/deeplearningzerotoall/TensorFlow.git 

If there were updates in this repository, you can now keep your local copy and your repository updated
저장소에 업데이트가 있는 경우 로컬 복사본과 repository 를 업데이트할 수 있습니다.
$ git pull upstream master && git push origin master 

Step 4: branch 만들기
You don't want to directly modify the master branch because the master branch keeps changing by merging PRs, etc.
Master branch는 Pull Request들을 계속 병합되고 수정되기 때문에 Master branch를 직접 수정하는 않는게 좋습니다. 
Also remember to give a meaningful name!
또한 의미 있는 이름으로 브랜치를 만드세요!
Example: 
예시:
$ git checkout -b hotfix/lab10 -t origin/master
After making a new branch, feel free to modify the codes now!
새로운 Branch를 만든 후에 자유롭게 코드를 수정하세요!
Note: don't get tempted to fix other things that are not related to your issue. Your commit should be in logical blocks! If it's a different problem, you have to create a separate issue.
//?? 
여러분의 Issue와 관련이 없는 다른 것들을 고치고 마세요! 여러분의 commit은 논리적으로 차곡차곡 쌓여야합니다. 만약 필요한 것이 다른 문제라면, 여러분은 또 다른 issue를 만들어야 합니다.
Step 5: Commit
If you have not set up, please set up your email/username
이메일/사용자 이름을 설정하세요.
$ git config --global user.name "Sung Kim"
$ git config --global user.email "sungkim@email.com"
then commit:
그리고 필요한 파일을 추가후, commit 하세요.
$ git add my/changed/files
$ git commit
Notes
주의사항
Write a clear commit message!
다른 사람들도 알아 볼 수 있게 명확한 Commit 메시지를 쓰세요
Example: 예시
Short (50 chars or less) summary of changes
 
More detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.
 
Further paragraphs come after blank lines.
 
  - Bullet points are okay, too
 
  - Typically a hyphen or asterisk is used for the bullet, preceded by a
	single space, with blank lines in between, but conventions vary here
Step 6: (선택사항) Rebase your branch
If your fix is taking longer than usual, it's likely that your repo is outdated.
Sync your repo to the latest:
만약 수정이 평소보다 더 오래 걸린다면, 레포지토리는 옛날 버전일 가능성이 있습니다. 항상 레포지토리를 최신 버전으로 동기화하세요.
$ git fetch upstream
$ git rebase upstream/master
Step 7: Push
Before pushing to YOUR REPO, make sure you run autopep8!
여러분의 repo를 push하기전에 ‘Autopep8’을 실행해주세요
Please follow PEP8 styles. The only exception is E501(max-line-char limit)
E501(최대 문자 줄 제한)을 제외한 모든 PEP8형식을 따라주세요.
Remember: Readability > everything
기억하기: 가독성이 최우선입니다
Example:
$ autopep8 . -r -i --ignore E501
$ git push -u origin hotfix/lab10
Step 8: Creating the PR
Now, if you open a browser and open this repo.
이제 여러분의 브라우저와 repo를 열면
You will see the big green button saying "compare & pull request."
"compare & pull request."라는 초록색 버튼을 보실 수 있습니다.
Please ensure you write a good title.
좋은 제목을 작성하세요.
Don't just write filenames you modified.
여러분이 수정한 파일이름을 쓰지마세요
Explain what you did and why you did.
여러분이 했던 것과 여러분이 왜 했었는지를 설명해주세요
Add a relevant issue number as well.
관련된 issue번호도 추가해주세요
Congratulations! Your PR will be reviewed by collaborators
축하합니다! 여러분의 PR은 collaborator들에게 검토받을겁니다.
Please check your PR pass the CI test as well.
여러분의 PR이 CI test도 통과했는지 체크하세요
 

