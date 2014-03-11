

package org.lemming.processors;

import org.lemming.data.Localization;
import org.lemming.data.XYLocalization;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;

import javax.swing.JOptionPane;

/**
 * Open a matlab session, feed it localizations. The process pause after launching a specified matlab program. Once the 
 * user is done, it gets localizations back (example of PALMsiever).
 * 
 * @author Joe Borbely, Thomas Pengo, Joran Deschamps
 */

public class MatlabSISOCommunicator extends SISO<Localization,Localization> {

	String path;									// Local path to the directory of the matlab program
	String matfile_name;							// Name of the matlab file .m (should be a command)

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
    
	int N=1;										// Counter for the localizations
	
	Boolean user_done = false;
	
	public MatlabSISOCommunicator(String s1, String s2) throws MatlabConnectionException, MatlabInvocationException{
		path = s1;
		matfile_name = s2;
		
		options = new MatlabProxyFactoryOptions.Builder()
        //.setHidden(b)			// Hide the main Matlab window
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);
		
	    proxy.eval("addpath "+path);											
		// the number of localizations is not known (memory reallocation)
		//proxy.eval("X=[]");
		//proxy.eval("xval=0");
		//proxy.eval("yval=0");
	}
	
	@Override
	public void process(Localization element) {
		
		// Add X and Y to a matlab vector
		try {
			proxy.setVariable("xval", element.getX());							// Buffer Matlab variable for x position
			proxy.setVariable("yval", element.getY());
			proxy.setVariable("n",N);											// Index of the localization
			proxy.eval("X(n,1)=xval;");											// Add to vector
			proxy.eval("Y(n,1)=yval;");
		} catch (MatlabInvocationException e1) {
			e1.printStackTrace();
		}
        
		N++;
		
		if(input.isEmpty()){
			
			try {
				this.runMatfile();												// Launch Matlab program
			} catch (MatlabInvocationException e) {
				e.printStackTrace();
			}
			
			// Wait until user is done
			JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
					null, "Waiting for end of MATLAB task" );
			
			user_done = true;
		    			
			// Get localizations back
		    try {
		    	double final_length = ((double[]) proxy.getVariable("length(X)"))[0];				//!!! can be done through matlabarray.getlength() or in java once X is rertieved
		    	//System.out.println(final_length);
				double[][] xloc = processor.getNumericArray("X").getRealArray2D();	// Matlab arrays are always 2D
				double[][] yloc = processor.getNumericArray("Y").getRealArray2D();

				// Send as outputs												// !!!!here what if palmsiever discard everything????
				for(int i=0;i<final_length;i++){
			    	//System.out.println(i);

					output.put(new XYLocalization(xloc[i][0],yloc[i][0]));   // !!!later might pass id if memory of it
				}															// !!!! what about using a workspace as output since the execution in matlab ends before 
		    } catch (MatlabInvocationException e) {
				e.printStackTrace();
			}

		    
			this.disconnectProxy();
		}
	}
	
	public void disconnectProxy(){
		proxy.disconnect();
	}
	
	void runMatfile() throws MatlabInvocationException{
		proxy.eval(matfile_name);
	}

	public boolean isUserDone() {
		return user_done;
	}

}
