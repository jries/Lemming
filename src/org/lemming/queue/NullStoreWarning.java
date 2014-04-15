package org.lemming.queue;

public class NullStoreWarning extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public NullStoreWarning(String name) {
		new ShowMessage("Error! The localization Store has not been set yet for "+name);
	}

}
