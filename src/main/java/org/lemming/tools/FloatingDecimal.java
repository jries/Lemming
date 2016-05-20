package org.lemming.tools;

import java.util.Arrays;

/**
 * Parser for the char buffered reader
 * 
 * @author Ronny Sczech
 *
 */
public class FloatingDecimal {

	private final boolean isNegative;
	private final int decExponent;
	private final char[] digits;
	private final int nDigits;
	private static final char zero[] = { '0', '0', '0', '0', '0', '0', '0', '0' };
	private static final int    maxDecimalDigits = 15;
	private static final int    intDecimalDigits = 9;
    private static final int    maxDecimalExponent = 308;
    private static final int    minDecimalExponent = -324;
    private static final int    bigDecimalExponent = 324;
	private static final double small10pow[] = {
	        1.0e0,
	        1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5,
	        1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10,
	        1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15,
	        1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20,
	        1.0e21, 1.0e22
	    };
	private static final double big10pow[] = {
	        1e16, 1e32, 1e64, 1e128, 1e256 };
	private static final double tiny10pow[] = {
	        1e-16, 1e-32, 1e-64, 1e-128, 1e-256 };
	
	private static final int maxSmallTen = small10pow.length-1;
	
	
	private FloatingDecimal( boolean negSign, int decExponent, char []digits, int n)
    {
        isNegative = negSign;
        this.decExponent = decExponent;
        this.digits = digits;
        this.nDigits = n;
    }
	
	private static char[] trim(char[] array) {
        int len = array.length;
        int st = 0;

        while ((st < len) && (array[st] <= ' ')) {
            st++;
        }
        while ((st < len) && (array[len - 1] <= ' ')) {
            len--;
        }
        return ((st > 0) || (len < array.length)) ? substring(array,st, len) : array;
    }
	
	private static char[] substring(char[] array, int beginIndex, int endIndex) {
		int subLen = endIndex - beginIndex;
        if (subLen < 0) 
            throw new IndexOutOfBoundsException();
		return Arrays.copyOfRange(array, beginIndex, endIndex);
	}

	public static FloatingDecimal read( char[] array ) throws NumberFormatException {
		 char    c;
		 boolean isNegative = false;
	     boolean signSeen   = false;
	     int decExp;
	     char[] in = trim(array);
	 
	 parseNumber:
        try{
        	int l = in.length;
            if ( l == 0 ) throw new NumberFormatException("empty String");
            int i = 0;
            switch ( in[i] ){
            case '-':
                isNegative = true;
                //FALLTHROUGH
            case '+':
                i++;
                signSeen = true;
            }
            //c = in[i];
            
            char[] digits = new char[ l ];
            int    nDigits= 0;
            boolean decSeen = false;
            int decPt = 0;
            int nLeadZero = 0;
            int nTrailZero= 0;
            
        digitLoop: 
        	while ( i < l ){
        		switch ( c = in[i] ){
                case '0':
                    if ( nDigits > 0 )
                        nTrailZero += 1;
                    else 
                        nLeadZero += 1;
                    break; // out of switch.
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    while ( nTrailZero > 0 ){
                        digits[nDigits++] = '0';
                        nTrailZero -= 1;
                    }
                    digits[nDigits++] = c;
                    break; // out of switch.
                case '.':
                    if ( decSeen ){
                        // already saw one ., this is the 2nd.
                        throw new NumberFormatException("multiple points");
                    }
                    decPt = i;
                    if ( signSeen ){
                        decPt -= 1;
                    }
                    decSeen = true;
                    break; // out of switch.
                default:
                    break digitLoop;
                }
                i++;
        	}
            
            if ( nDigits == 0 ){
                digits = zero;
                nDigits = 1;
                if ( nLeadZero == 0 ){
                    // we saw NO DIGITS AT ALL, not even a crummy 0!
                    // this is not allowed.
                    break parseNumber; // go throw exception
                }
            }
            
            if ( decSeen ){
                decExp = decPt - nLeadZero;
            } else {
                decExp = nDigits+nTrailZero;
            }
            
            /*
             * Look for 'e' or 'E' and an optionally signed integer.
             */
            if ( (i < l) &&  ((c = in[i] ) =='e' || c == 'E') ){
                int expSign = 1;
                int expVal  = 0;
                int reallyBig = Integer.MAX_VALUE / 10;
                boolean expOverflow = false;
                int expAt = i;
                switch( in[++i] ){
                case '-':
                    expSign = -1;
                    //FALLTHROUGH
                case '+':
                    i++;
                }
            expLoop:
                while ( i < l  ){
                    if ( expVal >= reallyBig ){
                        // the next character will cause integer
                        // overflow.
                        expOverflow = true;
                    }
                    switch ( c = in[i] ){
                    case '0':
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                        expVal = expVal*10 + ( c - '0' );
                        i++;
                        continue;
                    default:
                        i--;           // back up.
                        break expLoop; // stop parsing exponent.
                    }
                }
                int expLimit = bigDecimalExponent+nDigits+nTrailZero;
                if ( expOverflow || ( expVal > expLimit ) ){           
                    decExp = expSign*expLimit;
                } else {
                    decExp = decExp + expSign*expVal;
                }

                if ( i == expAt )
                    break parseNumber; // certainly bad
            }          
            
            if ( i+nTrailZero < l && i != l - 1) {
                break parseNumber; // go throw exception
            }
            
            return new FloatingDecimal( isNegative, decExp, digits, nDigits);        	
        } catch ( StringIndexOutOfBoundsException e )
        {
            System.out.println(e.getMessage());
        }
        throw new NumberFormatException("For input string: \"" + String.valueOf(in) + "\"");
	 }
	 
	 public strictfp double doubleValue(){
		
		 int     kDigits = Math.min( nDigits, maxDecimalDigits+1 );
		 long    lValue;
		 double  dValue;
		 double  rValue;
		 
		 // (special performance hack: start to do it using int)
         int iValue = digits[0]-'0';
         int iDigits = Math.min( kDigits, intDecimalDigits );
         for ( int i=1; i < iDigits; i++ ){
             iValue = iValue*10 + digits[i]-'0';
         }
         lValue = iValue;
         for ( int i=iDigits; i < kDigits; i++ ){
             lValue = lValue*10L + digits[i]-'0';
         }
         dValue = lValue;
         int exp = decExponent-kDigits;
		 
         if ( nDigits <= maxDecimalDigits ){
        	 
        	 if (exp == 0 || dValue == 0.0)
                 return (isNegative)? -dValue : dValue; // small floating integer
             else if ( exp >= 0 ){
                 if ( exp <= maxSmallTen ){
                     /*
                      * Can get the answer with one operation,
                      * thus one roundoff.
                      */
                     rValue = dValue * small10pow[exp];                 
                     return (isNegative)? -rValue : rValue;
                 }
                 int slop = maxDecimalDigits - kDigits;
                 if ( exp <= maxSmallTen+slop ){
                     /*
                      * We can multiply dValue by 10^(slop)
                      * and it is still "small" and exact.
                      * Then we can multiply by 10^(exp-slop)
                      * with one rounding.
                      */
                     dValue *= small10pow[slop];
                     rValue = dValue * small10pow[exp-slop];                   
                     return (isNegative)? -rValue : rValue;
                 }
                 /*
                  * Else we have a hard case with a positive exp.
                  */
             } else {
                 if ( exp >= -maxSmallTen ){
                     /*
                      * Can get the answer in one division.
                      */
                     rValue = dValue / small10pow[-exp];
                     return (isNegative)? -rValue : rValue;
                 }
                 /*
                  * Else we have a hard case with a negative exp.
                  */
             }
         }
         
         if ( exp > 0 ){
             if ( decExponent > maxDecimalExponent+1 ){
                 /*
                  * Lets face it. This is going to be
                  * Infinity. Cut to the chase.
                  */
                 return (isNegative)? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
             }
             if ( (exp&15) != 0 ){
                 dValue *= small10pow[exp&15];
             }
             if ( (exp>>=4) != 0 ){
                 int j;
                 for( j = 0; exp > 1; j++, exp>>=1 ){
                     if ( (exp&1)!=0)
                         dValue *= big10pow[j];
                 }
                 /*
                  * The reason for the weird exp > 1 condition
                  * in the above loop was so that the last multiply
                  * would get unrolled. We handle it here.
                  * It could overflow.
                  */
                 double t = dValue * big10pow[j];
                 if ( Double.isInfinite( t ) ){
                     /*
                      * It did overflow.
                      * Look more closely at the result.
                      * If the exponent is just one too large,
                      * then use the maximum finite as our estimate
                      * value. Else call the result infinity
                      * and punt it.
                      * ( I presume this could happen because
                      * rounding forces the result here to be
                      * an ULP or two larger than
                      * Double.MAX_VALUE ).
                      */
                     t = dValue / 2.0;
                     t *= big10pow[j];
                     if ( Double.isInfinite( t ) ){
                         return (isNegative)? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
                     }
                     t = Double.MAX_VALUE;
                 }
                 dValue = t;
             }
         } else if ( exp < 0 ){
             exp = -exp;
             if ( decExponent < minDecimalExponent-1 ){
                 /*
                  * Lets face it. This is going to be
                  * zero. Cut to the chase.
                  */
                 return (isNegative)? -0.0 : 0.0;
             }
             if ( (exp&15) != 0 ){
                 dValue /= small10pow[exp&15];
             }
             if ( (exp>>=4) != 0 ){
                 int j;
                 for( j = 0; exp > 1; j++, exp>>=1 ){
                     if ( (exp&1)!=0)
                         dValue *= tiny10pow[j];
                 }
                 /*
                  * The reason for the weird exp > 1 condition
                  * in the above loop was so that the last multiply
                  * would get unrolled. We handle it here.
                  * It could underflow.
                  */
                 double t = dValue * tiny10pow[j];
                 if ( t == 0.0 ){
                     /*
                      * It did underflow.
                      * Look more closely at the result.
                      * If the exponent is just one too small,
                      * then use the minimum finite as our estimate
                      * value. Else call the result 0.0
                      * and punt it.
                      * ( I presume this could happen because
                      * rounding forces the result here to be
                      * an ULP or two less than
                      * Double.MIN_VALUE ).
                      */
                     t = dValue * 2.0;
                     t *= tiny10pow[j];
                     if ( t == 0.0 ){
                         return (isNegative)? -0.0 : 0.0;
                     }
                     t = Double.MIN_VALUE;
                 }
                 dValue = t;
             }
         }

         return (isNegative)? -dValue : dValue;
	 }

}
