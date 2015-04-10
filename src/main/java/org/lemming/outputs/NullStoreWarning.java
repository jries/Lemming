package org.lemming.outputs;

/**
 * @author Ronny Sczech
 *
 */
public class NullStoreWarning extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * @param name - String
	 */
	public NullStoreWarning(String name) {
		new ShowMessage("Error! The localization Store has not been set yet for "+name);
	}

}
