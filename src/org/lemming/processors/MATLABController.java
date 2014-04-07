
package org.lemming.processors;

import javax.swing.JOptionPane;

import org.lemming.data.Localization;
import org.lemming.data.Workspace;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;


public class MATLABController<T1,T2> extends SISO<T1,T2>{

	String path;									// Local path to the directory of the matlab program
	String matfile_name;							// Name of the matlab file .m (should be a command)

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    //MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
    	
	//MATLABCommunicator2 communicator;
	
	Boolean user_done = false;
	
	public MATLABController(String s1, String s2) throws MatlabConnectionException{
		path = s1;
		matfile_name = s2;
		
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();

		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();
		
		//communicator = new MATLABCommunicator2(proxy);
		//communicator.input = this.input;					// must change
		//communicator.output = this.output;
	}
	
	@Override
	public void process(Object element) {
		//communicator.processInput(element);
		
		try {
			runMatfile();
		} catch (MatlabInvocationException e) {
			e.printStackTrace();
		}
		
		// Wait for user
		JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
				null, "Waiting for end of MATLAB task" );
		user_done = true;
		disconnectProxy();		
	}

	public void sendObject(Localization loc){
		
	}
	
	public void sendObject(Workspace workspace){
		
	}
	
	
	public void createProxy() throws MatlabConnectionException{
		if(!proxy.isConnected()){
			proxy = factory.getProxy();
		}
	}
	
	public void disconnectProxy(){
		if(proxy.isConnected()){
			proxy.disconnect();
		}		
	}
	
	public void changeMatfile(String path2, String matfile){
		path = path2;
		matfile_name = matfile;
	}
	
	public void runMatfile() throws MatlabInvocationException{									
		if(proxy.isConnected()){
			proxy.eval("addpath "+path);
			proxy.eval(matfile_name);
		}
	}
	
	public void sendCommand(String s) throws MatlabInvocationException{
		//communicator.sendCommand(s);
		proxy.eval(s);
	}
	
	public Boolean isUserDone(){
		return user_done;
	}
	
}