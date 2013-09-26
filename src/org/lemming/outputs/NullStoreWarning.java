package org.lemming.outputs;

public class NullStoreWarning extends RuntimeException {

	public NullStoreWarning(String name) {
		new ShowMessage("Error! The localization Store has not been set yet for "+name);
	}

}
