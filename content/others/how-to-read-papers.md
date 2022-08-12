---
title: "How to Read Papers"
date: 2022-08-11T19:21:20+08:00
draft: false
# markup: pdc
lastmod: [":fileModTime", ":default"]
categories:
    - 
tags:
    - 
description: ""
thumbnail: ""
keywords: ""
url: "how-to-read-papers"
---
The following workflow is a conclusion of the video: {{< youtube id="txjl_Q4jCyQ?list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I" >}}  from Mu Li of Amazon

```mermaid
flowchart TB
	next("another paper")
	subgraph p1 ["Pass 1(Selection)"]
		direction LR
		title[Title]
		abs[Abstract]
		conc[Conclusion]
		FaT["Figures and Tablets"]
		title-->abs-->conc--> FaT 
	end
	
	p1.-> dec{Decision}.->|"stop(low quality or\nlow relation)"| next
	dec.->|"go on"| p2
	subgraph p2 ["Pass 2(Selection)"]
		direction TB
		int_cit["Introduction:\nSome important Citations" ]
		method_ME["Method:\nwithout Math and Engineering\nfocus on figures and tablets"]	
		
	end
	
	p2.->dec2{Decision}
	dec2.->|"too hard to understand,\n go to one of\n the citations\nin introduction"| next
	dec2.->|"stop(not useful or\nnot need to\nunderstand all)"| next
	
	subgraph p3 ["Pass 3(Investigation)"]
		direction TB
		ew("Every single words of the paper\nand think about if I will do the\nsame things as the authors\ndid and what we can do next")
		code[Code it]
		
	end
	dec2 .->|"go on"| p3.->next
```

