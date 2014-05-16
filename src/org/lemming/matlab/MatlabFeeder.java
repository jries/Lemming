package org.lemming.matlab;

import org.lemming.data.HashWorkspace;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.data.XYFLocalization;
import org.lemming.outputs.SI;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;

/**
 * The MatlabFeeder is in charge to send data to an opened matlab session.
 * 
 * @author Joran Deschamps
 *
 */

public class MatlabFeeder<T> extends SI<T> {

	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
  
    int N; 											// Counter (of localizations)
    
	public MatlabFeeder(MatlabProxy p){
		N = 1; 											// In matlab counts begin at 1
		proxy = p;
		processor = new MatlabTypeConverter(proxy);
	}
	
	@Override
	public void process(T element) {
		
		// Test what the type of the element is
		
		
		/////////////////////////////////////////
		//// List of what need to be send:
		//// - (X,Y, frame)
		//// - hashworkspace
		////
		
		if(element instanceof XYFLocalization){
			// Add X and Y to a matlab vector
			try {
				proxy.setVariable("xval", ((XYFLocalization) element).getX());	    // Matlab buffer variable for x positions
				proxy.setVariable("yval", ((XYFLocalization) element).getY());
				proxy.setVariable("nframe", ((XYFLocalization) element).getFrame());
				proxy.setVariable("n",N);											// Index of the localization
				proxy.eval("X(n,1)=xval;");											// Add to vector
				proxy.eval("Y(n,1)=yval;");
				proxy.eval("F(n,1)=nframe;");
			} catch (MatlabInvocationException e1) {
				e1.printStackTrace();
			}
			
			N++;
			
		}else if(element instanceof Localization){
			// Add X and Y to a matlab vector
			try {
				proxy.setVariable("xval", ((Localization) element).getX());	    // Matlab buffer variable for x positions
				proxy.setVariable("yval", ((Localization) element).getY());
				proxy.setVariable("n",N);											// Index of the localization
				proxy.eval("X(n,1)=xval;");											// Add to vector
				proxy.eval("Y(n,1)=yval;");
			} catch (MatlabInvocationException e1) {
				e1.printStackTrace();
			}
			
			N++;
			
		} else if(element instanceof HashWorkspace){
				
				//processor.setNumericArray("loc", new MatlabNumericArray(img, null));
			
		}
		
	}
	
	public void reset(){
		N = 1;
	}

	
}
