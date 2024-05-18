# -*- coding: utf-8 -*-

def rk4(x, dx, y, deriv,params):
    """
      /*-----------------------------------------
      sous programme de resolution d'equations
      differentielles du premier ordre par
      la methode de Runge-Kutta d'ordre 4
      x = abscisse, une valeur scalaire, par exemple le temps
      dx = pas, par exemple le pas de temps
      y = valeurs des fonctions au temps t(i), c'est un tableau numpy de taille n
      avec n le nombre d'équations différentielles du 1er ordre
      
      rk4 renvoie les nouvelles valeurs de y pour t(i+1)
      
      deriv = variable contenant le nom du
      sous-programme qui calcule les derivees
      deriv doit avoir trois arguments: deriv(x,y,params) et renvoyer 
      un tableau numpy dy de taille n 
      ----------------------------------------*/
    """
    #  /* d1, d2, d3, d4 = estimations des derivees
    #    yp = estimations intermediaires des fonctions */  
    ddx = dx/2.       #         /* demi-pas */
    d1 = deriv(x,y,params)   #       /* 1ere estimation */          
    yp = y + d1*ddx
    #    for  i in range(n):
    #        yp[i] = y[i] + d1[i]*ddx
    d2 = deriv(x+ddx,yp,params)     #/* 2eme estimat. (1/2 pas) */
    yp = y + d2*ddx    
    d3 = deriv(x+ddx,yp,params)  #/* 3eme estimat. (1/2 pas) */
    yp = y + d3*dx    
    d4 = deriv(x+dx,yp,params)     #  /* 4eme estimat. (1 pas) */
    #/* estimation de y pour le pas suivant en utilisant
    #  une moyenne ponderee des derivees en remarquant
    #  que : 1/6 + 1/3 + 1/3 + 1/6 = 1 */
    return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6  