   c   l   e   a   r      
      
      
   L       =       6   4   ;      
   e   p   o   c   h       =       1   0   0   0   ;      
   l   o   a   d   (   "   t   o   r   c   h   _   o   u   t   p   u   t   _   "   +       e   p   o   c   h       +       "   .   m   a   t   "   )   ;      
      
   %   L       =       6   4   ;      
   %   l   o   a   d   (   "   .   .   /   n   e   t   w   o   r   k   _   o   u   t   p   u   t   /   p   r   e   t   r   a   i   n   _   t   e   s   t   /   t   o   r   c   h   _   o   u   t   p   u   t   _   L   _   "       +       L       +       "   s   e   e   d   _   "       +       0       +       "   .   m   a   t   "   )   ;      
   %   l   o   a   d   (   "   .   .   /   n   e   t   w   o   r   k   _   o   u   t   p   u   t   /   p   r   e   t   r   a   i   n   _   t   e   s   t   /   t   o   r   c   h   _   o   u   t   p   u   t   _   L   _   2   5   6   _   s   e   e   d   _   0   .   m   a   t   "   )      
      
   %   {      
   L   =       4   8      
   e   p   o   c   h       =       4   *   1   0   2   4   ;      
   l   o   a   d   (   "   .   .   /   n   e   t   w   o   r   k   _   o   u   t   p   u   t   /   L   _   s   w   e   e   p   /   t   o   r   c   h   _   o   u   t   p   u   t   _   L   _   "       +       L       +       "   _   e   p   o   c   h   _   "       +       e   p   o   c   h       +       "   .   m   a   t   "   )   ;      
   %   }      
      
   %   {      
   L   =       6   4   ;      
   e   p   o   c   h       =       4   *   1   0   2   4   ;      
   s   e   e   d       =       1   4   0   0   0   ;      
   l   o   a   d   (   "   .   .   /   n   e   t   w   o   r   k   _   o   u   t   p   u   t   /   s   e   e   d   _   s   w   e   e   p   /   t   o   r   c   h   _   o   u   t   p   u   t   _   L   _   "       +       L       +       "   _   e   p   o   c   h   _   "       +       e   p   o   c   h       +       "   _   s   e   e   d   _   "       +       s   e   e   d       +       "   .   m   a   t   "   )   ;      
   %   }      
      
   w       =       s   q   u   e   e   z   e   (   f   (   :   ,   :   ,   :   ,   1   )   )   ;      
      
   f   i   g   u   r   e   (   1   )      
   f   o   r       t       =       1   :   1   %   s   i   z   e   (   w   ,   3   )      
      
   t   i   l   e   d   l   a   y   o   u   t   (   2   ,   2   )   ;      
      
   n   e   x   t   t   i   l   e      
   w   0       =       s   q   u   e   e   z   e   (   w   (   1   :   e   n   d   -   1   ,   1   :   e   n   d   -   1   ,   t   )   .   '   )   ;      
   %   i   m   a   g   e   s   c   (   [   w   0   ,   w   0   ;   w   0   ,   w   0   ]       )   ;      
   i   m   a   g   e   s   c   (   w   0   )   ;      
   a   x   i   s       s   q   u   a   r   e   ;      
   s   e   t   (   g   c   a   ,   '   y   d   i   r   '   ,       '   n   o   r   m   a   l   '   )      
   t   i   t   l   e   (   t   )      
   c   o   l   o   r   m   a   p       b   l   u   e   w   h   i   t   e   r   e   d      
      
   n   e   x   t   t   i   l   e      
   s   e   m   i   l   o   g   y   (       l   o   s   s   _   h   i   s   t   o   r   y       )      
      
   n   e   x   t   t   i   l   e      
   s   c   a   t   t   e   r   (       x   s   (   :   ,   1   )   ,       x   s   (   :   ,   2   )   ,       '   f   i   l   l   e   d   '       )   ;      
   x   l   i   m   (   [   0   ,       2   *   p   i   ]   )      
   a   x   i   s       s   q   u   a   r   e      
   x   l   a   b   e   l   (   '   x   '   )   ;      
   y   l   a   b   e   l   (   '   y   '   )   ;      
   t   i   t   l   e   (   "   u   n   i   f   o   r   m       s   a   m   p   l   i   n   g   "   )   ;      
      
   n   e   x   t   t   i   l   e      
   x   s   _   N   N       =       m   o   d   (   x   s   _   N   N   ,       2   *   p   i   )   ;      
   s   c   a   t   t   e   r   (       x   s   _   N   N   (   :   ,   1   )   ,       x   s   _   N   N   (   :   ,   2   )   ,       '   f   i   l   l   e   d   '       )   ;      
   x   l   i   m   (   [   0   ,       2   *   p   i   ]   )      
   x   l   a   b   e   l   (   '   x   '   )   ;      
   y   l   a   b   e   l   (   '   y   '   )   ;      
   a   x   i   s       s   q   u   a   r   e      
   t   i   t   l   e   (   "   a   d   v   e   r   s   a   r   i   a   l       s   a   m   p   l   i   n   g   "   )   ;      
      
      
   d   r   a   w   n   o   w      
   e   n   d      
      
      
   %   %      
      
   f   o   r       p   r   e   s   e   e   d       =       1   :   1   4   0      
           s   e   e   d       =       p   r   e   s   e   e   d   *   1   0   0   0   ;      
           L   =       6   4   ;      
           e   p   o   c   h       =       4   *   1   0   2   4   ;      
           l   o   a   d   (   "   .   .   /   n   e   t   w   o   r   k   _   o   u   t   p   u   t   /   s   e   e   d   _   s   w   e   e   p   /   t   o   r   c   h   _   o   u   t   p   u   t   _   L   _   "       +       L       +       "   _   e   p   o   c   h   _   "       +       e   p   o   c   h       +       "   _   s   e   e   d   _   "       +       s   e   e   d       +       "   .   m   a   t   "   )   ;      
      
           c   l   f      
      
           w       =       s   q   u   e   e   z   e   (   f   (   :   ,   :   ,   :   ,   1   )   )   ;      
           f   i   g   u   r   e   (   1   )      
      
           w   0       =       s   q   u   e   e   z   e   (   w   (   1   :   e   n   d   -   1   ,   1   :   e   n   d   -   1   ,   t   )   .   '   )   ;      
           %   i   m   a   g   e   s   c   (   [   w   0   ,   w   0   ;   w   0   ,   w   0   ]       )   ;      
           i   m   a   g   e   s   c   (   w   0   )   ;      
           a   x   i   s       s   q   u   a   r   e   ;      
           s   e   t   (   g   c   a   ,   '   y   d   i   r   '   ,       '   n   o   r   m   a   l   '   )      
           c   o   l   o   r   m   a   p       b   l   u   e   w   h   i   t   e   r   e   d      
           t   i   t   l   e   (   "   s   e   e   d       =       "       +       s   e   e   d   )   ;      
      
           d   r   a   w   n   o   w      
      
           s   a   v   e   a   s   (       g   c   f   ,       "   s   e   e   d   _   s   w   e   e   p   /   "       +       s   e   e   d       +       "   .   p   n   g   "       )   ;      
   e   n   d