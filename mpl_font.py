import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#prop=fm.FontProperties(fname='C:/Windows/Fonts/NanumGothic.ttf')
prop=fm.FontProperties(fname='C:/Windows/Fonts/gulim.ttc')

plt.plot([1,2,3,4])
plt.title('테스트',fontproperties=prop,fontsize=20) # fontproperties 뒤에 fontsize 가 나와야 함
plt.legend(['하나'],prop=prop)
plt.show()

'''
p1,=plt.plot([1,2,3])
#p1.set_label('line')
p2,=plt.plot([2,3,4])
p3,=plt.plot([3,4,5])
plt.legend([p1,p3],['제곱함수','몰랑'],loc=3,prop=prop)
plt.show()
'''