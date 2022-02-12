# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# %%
def Exhaustive_search_all(template:np.ndarray,reference:np.ndarray):
    if template.shape[0] > reference.shape[0] or template.shape[1] > reference.shape[1]:
        print("The template is bigger than the reference image.")
        return None
    
    best_match = None
    best_match_score = float('-inf')
    for i in range(reference.shape[0] - template.shape[0] + 1):
        for j in range(reference.shape[1] - template.shape[1] + 1):
            score = ((template - reference[i:i+template.shape[0],j:j+template.shape[1]])**2).sum()

            if score >= best_match_score:
                best_match_score = score
                best_match = (i,j)

    return best_match, best_match_score

# %%
def exhaustive_search(template:np.ndarray,reference:np.ndarray,X:int,Y:int,p:int):
    if template.shape[0] > reference.shape[0] or template.shape[1] > reference.shape[1]:
        print("The template is bigger than the reference image.")
        return None
    
    best_match = None
    best_match_score = float('-inf')
    count = 0
    for x in range(X-p,X+p+1):
        for y in range(Y-p,Y+p+1):
                
            if x < 0 or x+template.shape[0] > reference.shape[0] or y < 0 or y+template.shape[1] > reference.shape[1]:
                continue
            
            if x >= reference.shape[0] or y >= reference.shape[1]:
                continue

            score = ((template - reference[x:x+template.shape[0],y:y+template.shape[1]])**2).sum()
            count += 1
            if score >= best_match_score:
                best_match_score = score
                best_match = (x,y)

    return best_match, best_match_score , count

# %%
def logarithomic_search(template:np.ndarray,reference:np.ndarray,X:int,Y:int,p:int):
    best_match = None
    best_match_score = float('-inf')

    level = p
    count = 0
    completed = set()

    while level > 0:
        x_cors = [-level//2,0,-(level // -2)]
        y_cors = [-level//2,0,-(level // -2)]
        level = level // 2

        for x in x_cors:
            for y in y_cors:

                if X+x < 0 or X+x+template.shape[0] > reference.shape[0] or Y+y < 0 or Y+y+template.shape[1] > reference.shape[1]:
                    continue
                
                if X+x >= reference.shape[0] or Y+y >= reference.shape[1]:
                    continue

                if (X+x,Y+y) in completed:
                    continue

                score = ((template - reference[X+x:X+x+template.shape[0],Y+y:Y+y+template.shape[1]])**2).sum()
                count+=1

                if score >= best_match_score:
                    best_match_score = score
                    best_match = (X+x,Y+y)
                
                completed.add((X+x,Y+y))

        X,Y = best_match

    return best_match, best_match_score , count
        
    

# %%
def heiararchical_search(template:np.ndarray,reference:np.ndarray,X:int,Y:int,p:int,level=2,func=exhaustive_search):
    
    filtered_reference = []
    filtered_template = []

    filtered_reference.append(reference)
    filtered_template.append(template)

    for _ in range(level):
        filtered_reference.append(cv2.pyrDown(filtered_reference[-1]))
        filtered_template.append(cv2.pyrDown(filtered_template[-1]))

    best_match = None
    best_match_score = None
    best_match_prev = (X//2**level,Y//2**level)

    count = 0

    for i in range(level,-1,-1):

        best_match,best_match_score, cnt = func(filtered_template[i],filtered_reference[i],best_match_prev[0],best_match_prev[1],p//2**i)
        count += cnt
        best_match_prev = (best_match[0]*2,best_match[1]*2)

    
    return best_match, best_match_score, count
        
    

# %%
def template_matching(frames:np.ndarray,template:np.ndarray,p:int=50,func=exhaustive_search):
    best_matches = []
    best_match, best_match_score = Exhaustive_search_all(template,frames[0])
    counts = []
    for i in range(len(frames)):
        best_match , best_match_score, cnt = func(template,frames[i],best_match[0],best_match[1],p)
        best_matches.append(best_match)
        counts.append(cnt)
    
    return best_matches, np.mean(counts)


# %%
def draw_rect(frames:np.ndarray,best_matches:list,template:np.ndarray,width:int=5):
    rect_frames = frames.copy()
    for i in range(len(frames)):
        cv2.rectangle(rect_frames[i],(best_matches[i][1],best_matches[i][0]),(best_matches[i][1]+template.shape[1],best_matches[i][0]+template.shape[0]),(255,0,0),width)
    return rect_frames

# %%
def make_movie(frames:list,file_name:str,fps:int):
    video_mov = cv2.VideoWriter(filename=file_name,fourcc=cv2.VideoWriter_fourcc(*'movv'),fps=fps,frameSize=(frames[0].shape[1],frames[0].shape[0]))
    for i in range(len(frames)):
        video_mov.write(frames[i])
    video_mov.release()
    cv2.destroyAllWindows()


# %%
def make_report(frames:list,template:np.ndarray):
    p_list = list(range(2,15,3))

    report = []
    for p in p_list:
        _,count_hi = template_matching(frames,template,p,func=heiararchical_search)
        _,count_ex = template_matching(frames,template,p,func=exhaustive_search)
        _,count_lo = template_matching(frames,template,p,func=logarithomic_search)

        report.append([p,count_hi,count_ex,count_lo])

    repot_df = pd.DataFrame(np.array(report),columns=['p' , 'Exhaustive' , '2D Log' ,  'Hierarchical'])
    repot_df.to_csv('1605104.csv')
    
    return repot_df

# %%
def make_all_videos(frames:list,buf:np.ndarray,template:np.ndarray,fps,p:int=10):

    for i in [heiararchical_search,exhaustive_search,logarithomic_search]:
        best_matches,count = template_matching(frames,template,p,func=i)
        print(i.__name__,count)
        rect_frames = draw_rect(buf,best_matches,template)
        make_movie(rect_frames,'1605104_{}.mov'.format(i.__name__),fps)

# %%
cap = cv2.VideoCapture('input.mov')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
grey_scale = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    grey_scale[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2GRAY)
    fc += 1

cap.release()

template = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)

# %%
# report = make_report(grey_scale,template)

# %%
make_all_videos(grey_scale,buf,template,fps)


