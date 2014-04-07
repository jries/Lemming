package org.lemming.utils;

import javax.swing.JOptionPane;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;

public class MatlabOpener implements Runnable {

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
	
	String path;
	String matcommand;
	Boolean user_done = false;
	
	public MatlabOpener(String s1, String s2) throws MatlabConnectionException, MatlabInvocationException{
		path = s1;
		matcommand = s2;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	}
	
	public void runMatfile(){
		try {
			proxy.eval("addpath "+path);
			proxy.eval(matcommand);
		} catch (MatlabInvocationException e) {
			e.printStackTrace();
		}
			
		// Wait until user is done
		JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
		 	null, "Waiting for end of MATLAB task" );

		user_done = true;
	}
	
	public void disconnectProxy(){
		proxy.disconnect();
	}
	
	public boolean isUserDone() {
		return user_done;
	}

	@Override
	public void run() {
		runMatfile();
		disconnectProxy();		
	}
	
}
