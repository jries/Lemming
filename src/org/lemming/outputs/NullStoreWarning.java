package org.lemming.outputs;

public class NullStoreWarning {

	public NullStoreWarning(String name) {
		new ShowMessage("Error! The localization Store has not been set yet for "+name);
	}

}
