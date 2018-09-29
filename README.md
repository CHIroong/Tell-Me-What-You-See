# SoundGlance: Briefing the Visual Cues of Web Pages for Screen Reader Users

> Screen readers have become a core assistive technology for blind web users to explore and navigate web pages. However, screen readers only convey the textual information or structural properties of web pages, such as the number of headings, not their overall impression. Such a limitation delays blind web users from judging the relevance and credibility of web pages, which non-blind people can do immediately by skimming the web pages. As such, we present SoundGlance, a novel sonification application that briefly delivers an auditory summary of web pages. SoundGlance supports blind web users in evaluating web pages by sonifying the important visual cues of the pages--the amount of advertisements and visually salient words--along with the summarized content.

## Motivation

시각장애인은 스크린 리더를 통해 웹을 탐색합니다. 그러나 아무리 숙련자라도 페이지 하나의 구성 정보를 파악하는 데 약 1~2분이 소요됩니다.
이는 웹 서핑의 효율성을 크게 저해한다고 생각되었고, 저희는 약 15초 이내 길이의 "웹 페이지 미리보기"를 자동 추출하는 프로그램을 만들었습니다.

## Design

SoundGlance의 15초 미리보기는 다음과 같은 세 구성요소로 이루어져 있습니다.

### 1. 텍스트, 이미지, 광고 비율

웹 페이지의 스크린샷과 DOM 정보로부터 이를 구성하는 텍스트 / 이미지 / 광고 영역의 넓이를 CNN을 통해 예측한 뒤 비프음으로 전달합니다.

* 최대 길이:  5초
* 텍스트 영역 넓이: 낮은 단음
* 이미지 영역 넓이: 중간 화음
* 광고 영역 넓이: 높은 불협화음

### 2. 시각적으로 강조된 문장

페이지에서 가장 크기가 큰 순서대로 문장을 추출해서 최대 5초 동안 읽습니다.

### 3. 요약된 내용

[LexRank](https://github.com/theeluwin/lexrankr) 알고리즘을 이용해 3문장으로 요약된 페이지 본문 내용을 최대 5초 동안 읽습니다.


## Evaluation

* 6개의 검색 상황 및 검색 키워드를 결정하고, 각 키워드로부터 8개의 웹사이트를 추출했습니다.
* 8개의 웹사이트는 (신뢰도가 높음/낮음, 관련성이 높음/낮음)의 4가지 카테고리 각각 2개씩 입니다. 3명의 연구자가 신뢰도, 관련성을 평가했습니다.
* SoundGlance가 추출한 15초 미리보기를 제공하는 경우와, 스크린 리더 사용자가 일반적으로 웹을 탐색하는 방법인 <title>, <nav>, <h>를 15초 동안 읽어주는 경우의 효과를 비교했습니다.
* 미리보기만 듣고 해당 페이지를 더 탐색할 가치가 있는지 물어보았고, 연구자가 사전에 평정한 신뢰도 및 관련성과의 상관이 있기를 기대했지만 두 조건 간 유의미한 차이는 발견되지 않았습니다.

## Limitation

* 제작 동기에 따르면 real-time, online으로 추출 및 청각화가 이루어져야 하지만 아직 그러지 못합니다. (페이지 하나 추출에 약 5분 정도 걸림)
* 기획 단계에서 실제 스크린 리더 사용자로부터의 의견을 듣는 일을 소홀히 하여 "익숙하지 않고 유용성이 떨어진다"는 평을 많이 받았습니다.
