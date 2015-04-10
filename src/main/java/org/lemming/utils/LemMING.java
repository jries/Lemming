package org.lemming.utils;

/**
 * @author Ronny Sczech
 *
 */
public class LemMING {
	
	/**
	 * Throws a {@code RuntimeException} with the specified error message.
	 * 
	 * @param errMessage - the error message to display
	 */
	public static void error(String errMessage) {
		System.err.println(errMessage);
		
		throw new RuntimeException(errMessage);
	}

	/**
	 * Display a warning message.
	 * 
	 * @param warningMessage - the warning message to display
	 */
	public static void warning(String warningMessage) {
		System.out.println(warningMessage);
	}

	/**
	 * Add a pause to the program.<br><br>
	 * E.g., allow for an image to be displayed for the specified amount of time
	 * 
	 * @param msec - the time, in milliseconds, that the program should sleep
	 */
	public static void pause(long msec) {
		try {
			Thread.sleep(msec); 
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		} 		
	}

}